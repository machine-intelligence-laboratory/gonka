package poc

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"decentralized-api/chainphase"
	"decentralized-api/cosmosclient"
	"decentralized-api/logging"
	"decentralized-api/poc/artifacts"
	"decentralized-api/poc/propagation"

	"github.com/productscience/inference/api/inference/inference"
	"github.com/productscience/inference/x/inference/types"
)

const distributionRetryInterval = 30 * time.Second
const DefaultObservationBuffer = int64(10)

type commitState struct {
	count    uint32
	rootHash []byte
}

type CommitWorker struct {
	store              *artifacts.ManagedArtifactStore
	recorder           cosmosclient.CosmosMessageClient
	tracker            *chainphase.ChainPhaseTracker
	participantAddress string
	pubKey             string

	interval time.Duration
	stop     chan struct{}
	done     chan struct{}

	mu                      sync.Mutex
	currentPocHeight        int64
	lastDistributionAttempt time.Time
	lastCommitted           map[int64]commitState

	propagationEnabled   bool
	bundler              *propagation.Bundler
	propagationCache     *propagation.Cache
	observationSubmitted map[int64]bool
	consensusSubmitted   map[int64]bool
}

func NewCommitWorker(
	store *artifacts.ManagedArtifactStore,
	recorder cosmosclient.CosmosMessageClient,
	tracker *chainphase.ChainPhaseTracker,
	participantAddress string,
	pubKey string,
	interval time.Duration,
	propagationEnabled bool,
	bundler *propagation.Bundler,
	propagationCache *propagation.Cache,
) *CommitWorker {
	w := &CommitWorker{
		store:                store,
		recorder:             recorder,
		tracker:              tracker,
		participantAddress:   participantAddress,
		pubKey:               pubKey,
		interval:             interval,
		stop:                 make(chan struct{}),
		done:                 make(chan struct{}),
		lastCommitted:        make(map[int64]commitState),
		propagationEnabled:   propagationEnabled,
		bundler:              bundler,
		propagationCache:     propagationCache,
		observationSubmitted: make(map[int64]bool),
		consensusSubmitted:   make(map[int64]bool),
	}

	store.StartPeriodicFlush(interval)

	go w.run()
	logging.Info("CommitWorker started", types.PoC, "interval", interval)
	return w
}

func (w *CommitWorker) Close() {
	close(w.stop)
	<-w.done
	w.store.StopPeriodicFlush()
	logging.Info("CommitWorker stopped", types.PoC)
}

func (w *CommitWorker) run() {
	defer close(w.done)
	ticker := time.NewTicker(w.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			w.tick()
		case <-w.stop:
			return
		}
	}
}

func (w *CommitWorker) tick() {
	epochState := w.tracker.GetCurrentEpochState()
	if epochState == nil || !epochState.IsSynced {
		return
	}

	if !ShouldUseV2FromEpochState(epochState) {
		return
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	pocHeight := GetCurrentPocStageHeight(epochState)

	if pocHeight > 0 && w.currentPocHeight != pocHeight {
		w.currentPocHeight = pocHeight
		w.lastDistributionAttempt = time.Time{}
		w.lastCommitted = make(map[int64]commitState)
		w.observationSubmitted = make(map[int64]bool)
		w.consensusSubmitted = make(map[int64]bool)
	}

	if pocHeight > 0 {
		isPoCPhase := epochState.CurrentPhase == types.PoCGeneratePhase ||
			epochState.CurrentPhase == types.PoCGenerateWindDownPhase
		canCommit := ShouldAcceptStoreCommit(epochState, pocHeight)
		logging.Debug("CommitWorker: tick", types.PoC,
			"phase", epochState.CurrentPhase,
			"pocHeight", pocHeight,
			"isPoCPhase", isPoCPhase,
			"canCommit", canCommit)
		if isPoCPhase {
			w.maybePublishHeaders(pocHeight)
		}
		if canCommit {
			w.maybeSubmitObservation(pocHeight)
			w.maybeSubmitConsensusCommit(pocHeight)
		}
	}

	if ShouldHaveDistributedWeights(epochState) && pocHeight > 0 {
		shouldRetry := w.lastDistributionAttempt.IsZero() ||
			time.Since(w.lastDistributionAttempt) > distributionRetryInterval
		onChain := w.isDistributionOnChain(pocHeight)
		logging.Debug("CommitWorker: distribution check", types.PoC,
			"pocHeight", pocHeight,
			"shouldRetry", shouldRetry,
			"lastAttemptIsZero", w.lastDistributionAttempt.IsZero(),
			"onChain", onChain)
		if shouldRetry && !onChain {
			w.submitWeightDistribution(pocHeight)
		}
	}
}

func (w *CommitWorker) maybePublishHeaders(pocHeight int64) {
	if !w.propagationEnabled || w.bundler == nil {
		return
	}

	store, err := w.store.GetStore(pocHeight)
	if err != nil || store == nil {
		return
	}

	count, rootHash := store.GetFlushedRoot()
	if count == 0 || rootHash == nil {
		return
	}

	last := w.lastCommitted[pocHeight]
	if last.count == count && bytes.Equal(last.rootHash, rootHash) {
		return
	}

	bundleID := propagation.MakeBundleID(w.participantAddress, pocHeight, rootHash, count)

	if err := w.bundler.Publish(pocHeight, w.participantAddress, w.pubKey, count, rootHash); err != nil {
		logging.Warn("CommitWorker: propagation header publish failed", types.PoC,
			"pocHeight", pocHeight, "error", err)
	} else {
		logging.Info("CommitWorker: header published via propagation", types.PoC,
			"pocHeight", pocHeight, "count", count)

		if err := w.bundler.StoreOwnArrival(pocHeight, w.participantAddress, count); err != nil {
			logging.Warn("CommitWorker: failed to store own arrival", types.PoC,
				"pocHeight", pocHeight, "error", err)
		}

		if err := w.publishProofsViaPropagation(store, bundleID, count); err != nil {
			logging.Warn("CommitWorker: propagation proofs publish failed", types.PoC,
				"pocHeight", pocHeight, "error", err)
		}
	}
	w.lastCommitted[pocHeight] = commitState{count, rootHash}
}

func (w *CommitWorker) maybeSubmitObservation(pocHeight int64) {
	if !w.propagationEnabled || w.propagationCache == nil {
		return
	}

	if w.observationSubmitted[pocHeight] {
		return
	}

	arrivals, err := w.propagationCache.GetAllFirstArrivals(pocHeight)
	if err != nil || len(arrivals) == 0 {
		return
	}

	protoArrivals := make([]*inference.PoCObservationArrival, 0, len(arrivals))
	for participant, info := range arrivals {
		if info.Count > 0 {
			protoArrivals = append(protoArrivals, &inference.PoCObservationArrival{
				Participant: participant,
				Count:       info.Count,
			})
		}
	}

	if len(protoArrivals) == 0 {
		return
	}

	msg := &inference.MsgSubmitPoCObservation{
		PocStageStartBlockHeight: pocHeight,
		Arrivals:                 protoArrivals,
	}

	if err := w.recorder.SubmitPoCObservation(msg); err != nil {
		logging.Warn("CommitWorker: observation submission failed", types.PoC,
			"pocHeight", pocHeight, "error", err)
		return
	}

	w.observationSubmitted[pocHeight] = true
	logging.Info("CommitWorker: observation submitted on-chain", types.PoC,
		"pocHeight", pocHeight, "arrivals", len(protoArrivals))
}

func (w *CommitWorker) maybeSubmitConsensusCommit(pocHeight int64) {
	if w.consensusSubmitted[pocHeight] {
		return
	}

	store, err := w.store.GetStore(pocHeight)
	if err != nil || store == nil {
		logging.Debug("CommitWorker: no store for height", types.PoC, "pocHeight", pocHeight)
		return
	}

	count, rootHash := store.GetFlushedRoot()
	if rootHash == nil {
		logging.Debug("CommitWorker: no flushed data", types.PoC, "pocHeight", pocHeight)
		return
	}

	if !w.propagationEnabled {
		if count == 0 {
			return
		}

		msg := &inference.MsgPoCV2StoreCommit{
			PocStageStartBlockHeight: pocHeight,
			Count:                    count,
			RootHash:                 rootHash,
		}

		if err := w.recorder.SubmitPoCV2StoreCommit(msg); err != nil {
			logging.Warn("CommitWorker: local commit failed", types.PoC,
				"pocHeight", pocHeight, "error", err)
			return
		}

		w.consensusSubmitted[pocHeight] = true
		logging.Info("CommitWorker: local count committed (propagation disabled)", types.PoC,
			"pocHeight", pocHeight, "count", count)
		return
	}

	queryClient := w.recorder.NewInferenceQueryClient()
	resp, err := queryClient.PoCConsensus(context.Background(), &types.QueryPoCConsensusRequest{
		PocStageStartBlockHeight: pocHeight,
	})
	if err != nil {
		logging.Warn("CommitWorker: failed to query on-chain consensus", types.PoC,
			"pocHeight", pocHeight, "error", err)
		return
	}

	var agreedCount uint32
	for _, entry := range resp.Entries {
		if entry.Participant == w.participantAddress && entry.AgreedCount > 0 {
			agreedCount = entry.AgreedCount
			logging.Info("CommitWorker: on-chain consensus reached", types.PoC,
				"pocHeight", pocHeight, "agreedCount", agreedCount,
				"validators", entry.TotalValidators, "agreeing", entry.AgreeingCount)
			break
		}
	}

	if agreedCount > 0 {
		commitRootHash := rootHash
		if agreedCount != count {
			commitRootHash, err = store.GetRootAt(agreedCount)
			if err != nil {
				logging.Warn("CommitWorker: failed to get root at consensus count", types.PoC,
					"pocHeight", pocHeight, "agreedCount", agreedCount, "error", err)
				return
			}
		}

		msg := &inference.MsgPoCV2StoreCommit{
			PocStageStartBlockHeight: pocHeight,
			Count:                    agreedCount,
			RootHash:                 commitRootHash,
		}

		if err := w.recorder.SubmitPoCV2StoreCommit(msg); err != nil {
			logging.Warn("CommitWorker: consensus commit failed", types.PoC,
				"pocHeight", pocHeight, "error", err)
			return
		}

		w.consensusSubmitted[pocHeight] = true
		logging.Info("CommitWorker: consensus committed", types.PoC,
			"pocHeight", pocHeight, "agreedCount", agreedCount)

		if w.propagationEnabled && w.bundler != nil && agreedCount != count {
			newBundleID := propagation.MakeBundleID(w.participantAddress, pocHeight, commitRootHash, agreedCount)
			if err := w.publishProofsViaPropagation(store, newBundleID, agreedCount); err != nil {
				logging.Warn("CommitWorker: failed to re-publish proofs at consensus count", types.PoC,
					"pocHeight", pocHeight, "agreedCount", agreedCount, "error", err)
			}
		}
		return
	}

	logging.Debug("CommitWorker: waiting for on-chain consensus, will retry", types.PoC,
		"pocHeight", pocHeight)
}

func (w *CommitWorker) publishProofsViaPropagation(store *artifacts.ArtifactStore, bundleID [32]byte, count uint32) error {
	if count == 0 {
		return nil
	}

	proofs := make([]propagation.ProofItem, 0, count)

	for leafIndex := uint32(0); leafIndex < count; leafIndex++ {
		nonce, vector, err := store.GetArtifact(leafIndex)
		if err != nil {
			logging.Warn("CommitWorker: failed to get artifact for propagation", types.PoC,
				"leafIndex", leafIndex, "error", err)
			continue
		}

		proof, err := store.GetProof(leafIndex, count)
		if err != nil {
			logging.Warn("CommitWorker: failed to get proof for propagation", types.PoC,
				"leafIndex", leafIndex, "error", err)
			continue
		}

		proofStrings := make([]string, len(proof))
		for i, hash := range proof {
			proofStrings[i] = base64.StdEncoding.EncodeToString(hash)
		}

		proofs = append(proofs, propagation.ProofItem{
			LeafIndex:   leafIndex,
			NonceValue:  nonce,
			VectorBytes: base64.StdEncoding.EncodeToString(vector),
			Proof:       proofStrings,
		})
	}

	if len(proofs) > 0 {
		if err := w.bundler.PublishProofs(bundleID, proofs); err != nil {
			return err
		}
		logging.Info("CommitWorker: proofs published via propagation", types.PoC,
			"bundleID", fmt.Sprintf("%x", bundleID[:8]), "proofCount", len(proofs))
	}

	return nil
}

func (w *CommitWorker) isDistributionOnChain(pocHeight int64) bool {
	if w.participantAddress == "" {
		return false
	}
	queryClient := w.recorder.NewInferenceQueryClient()
	resp, err := queryClient.MLNodeWeightDistribution(context.Background(), &types.QueryMLNodeWeightDistributionRequest{
		PocStageStartBlockHeight: pocHeight,
		ParticipantAddress:       w.participantAddress,
	})
	return err == nil && resp.Found
}

func (w *CommitWorker) submitWeightDistribution(pocHeight int64) {
	store, err := w.store.GetStore(pocHeight)
	if err != nil || store == nil {
		logging.Debug("CommitWorker: no store", types.PoC, "pocHeight", pocHeight)
		return
	}

	if w.participantAddress == "" {
		logging.Debug("CommitWorker: no participant address", types.PoC)
		return
	}

	queryClient := w.recorder.NewInferenceQueryClient()
	resp, err := queryClient.PoCV2StoreCommit(context.Background(), &types.QueryPoCV2StoreCommitRequest{
		PocStageStartBlockHeight: pocHeight,
		ParticipantAddress:       w.participantAddress,
	})
	if err != nil {
		logging.Warn("CommitWorker: failed to query last commit", types.PoC,
			"pocHeight", pocHeight, "error", err)
		return
	}
	if !resp.Found || resp.Count == 0 {
		logging.Debug("CommitWorker: no committed snapshot", types.PoC,
			"pocHeight", pocHeight, "found", resp.Found, "count", resp.Count)
		return
	}

	if err := store.Flush(); err != nil {
		logging.Warn("CommitWorker: flush failed", types.PoC, "pocHeight", pocHeight, "error", err)
	}

	distribution := store.GetNodeDistribution()
	if len(distribution) == 0 {
		logging.Debug("CommitWorker: empty distribution", types.PoC, "pocHeight", pocHeight)
		return
	}

	weights, err := getWeightDistribution(distribution, resp.Count)
	if err != nil {
		logging.Error("CommitWorker: failed to build weight distribution", types.PoC,
			"pocHeight", pocHeight, "error", err)
		return
	}

	msg := &inference.MsgMLNodeWeightDistribution{
		PocStageStartBlockHeight: pocHeight,
		Weights:                  weights,
	}

	if err := w.recorder.SubmitMLNodeWeightDistribution(msg); err != nil {
		logging.Warn("CommitWorker: distribution failed", types.PoC,
			"pocHeight", pocHeight, "error", err)
		return
	}

	w.lastDistributionAttempt = time.Now()

	logging.Info("CommitWorker: distributed weights", types.PoC,
		"pocHeight", pocHeight, "nodes", len(weights), "count", resp.Count,
		"distribution", formatWeightDistribution(weights))
}

func getWeightDistribution(distribution map[string]uint32, targetCount uint32) ([]*inference.MLNodeWeight, error) {
	if len(distribution) == 0 {
		return nil, fmt.Errorf("empty distribution")
	}
	if targetCount == 0 {
		return nil, fmt.Errorf("targetCount is 0")
	}

	var localSum uint32
	for _, count := range distribution {
		localSum += count
	}

	if localSum == 0 {
		return nil, fmt.Errorf("distribution sum is 0")
	}

	if localSum == targetCount {
		weights := make([]*inference.MLNodeWeight, 0, len(distribution))
		for nodeId, count := range distribution {
			weights = append(weights, &inference.MLNodeWeight{
				NodeId: nodeId,
				Weight: count,
			})
		}
		return weights, nil
	}

	logging.Warn("CommitWorker: adjusting distribution proportionally", types.PoC,
		"localSum", localSum, "targetCount", targetCount)

	ratio := float64(targetCount) / float64(localSum)

	keys := make([]string, 0, len(distribution))
	for nodeId := range distribution {
		keys = append(keys, nodeId)
	}
	sort.Strings(keys)

	weights := make([]*inference.MLNodeWeight, 0, len(distribution))
	var scaledSum uint32
	for _, nodeId := range keys {
		count := distribution[nodeId]
		scaled := uint32(float64(count) * ratio)
		weights = append(weights, &inference.MLNodeWeight{
			NodeId: nodeId,
			Weight: scaled,
		})
		scaledSum += scaled
	}

	diff := int(targetCount) - int(scaledSum)
	for i := 0; diff > 0; i++ {
		weights[i%len(weights)].Weight++
		diff--
	}

	return weights, nil
}

func formatWeightDistribution(weights []*inference.MLNodeWeight) string {
	if len(weights) == 0 {
		return "{}"
	}
	parts := make([]string, len(weights))
	for i, w := range weights {
		parts[i] = fmt.Sprintf("%s:%d", w.NodeId, w.Weight)
	}
	return "{" + strings.Join(parts, ", ") + "}"
}
