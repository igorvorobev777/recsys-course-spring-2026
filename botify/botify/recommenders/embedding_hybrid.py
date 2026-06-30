import pickle
import joblib
import json
import random
from collections import defaultdict

from .recommender import Recommender


class EmbeddingHybridRecommender(Recommender):
    def __init__(self, track_redis, catalog, sasrec_redis, lightfm_redis, listen_history_redis,
                 fallback_recommender: Recommender, embeddings_path: str):
        self.track_redis = track_redis
        self.catalog = catalog
        self.sasrec_redis = sasrec_redis
        self.lightfm_redis = lightfm_redis
        self.listen_history_redis = listen_history_redis
        self.fallback_recommender = fallback_recommender

        self.track_embeddings = joblib.load(embeddings_path)

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_user_history(user)
        if not history:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)
        seen_tracks = {t for t, _ in history}

        history_artist_counts = defaultdict(int)
        for track_id, _ in history:
            track_bytes = self.track_redis.get(track_id)
            if track_bytes is None:
                continue
            track = self.catalog.from_bytes(track_bytes)
            artist_id = track.artist_id
            if artist_id is not None:
                history_artist_counts[artist_id] += 1

        anchor_id = self._select_anchor(history, prev_track)
        anchor_vec = self.track_embeddings.get(anchor_id)
        if anchor_vec is None:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        sasrec_raw = self.sasrec_redis.get(anchor_id)
        lightfm_raw = self.lightfm_redis.get(anchor_id)

        sasrec_cands = [int(t) for t in pickle.loads(sasrec_raw)] if sasrec_raw else []
        lightfm_cands = [int(t) for t in pickle.loads(lightfm_raw)] if lightfm_raw else []

        if not sasrec_cands and not lightfm_cands:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        RRF_K = 60

        sasrec_ranks = {t: i for i, t in enumerate(sasrec_cands)}
        lightfm_ranks = {t: i for i, t in enumerate(lightfm_cands)}

        all_tracks = set(sasrec_ranks.keys()).union(set(lightfm_ranks.keys()))

        rrf_scores = {}
        for t in all_tracks:
            score = 0.0
            if t in sasrec_ranks:
                score += 1.0 / (RRF_K + sasrec_ranks[t])
            if t in lightfm_ranks:
                score += 1.0 / (RRF_K + lightfm_ranks[t])
            rrf_scores[t] = score

        all_candidates = sorted(rrf_scores.keys(), key=lambda t: rrf_scores[t], reverse=True)

        candidates = [t for t in all_candidates if t not in seen_tracks]
        if not candidates:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        num_cands = len(candidates)

        SEMANTIC_WEIGHT = 0.65
        RANK_WEIGHT = 0.35
        ARTIST_PENALTY_COEFF = 0.05

        best_track = None
        max_score = -float('inf')

        for i, cand_id in enumerate(candidates):
            cand_vec = self.track_embeddings.get(cand_id)
            if cand_vec is None:
                continue

            semantic_score = sum(float(a) * float(c) for a, c in zip(anchor_vec, cand_vec))
            rank_score = 1.0 - (i / (num_cands - 1)) if num_cands > 1 else 1.0

            cand_artist_id = None
            track_bytes = self.track_redis.get(cand_id)
            if track_bytes is not None:
                track = self.catalog.from_bytes(track_bytes)
                cand_artist_id = track.artist_id

            penalty = history_artist_counts.get(cand_artist_id, 0) * ARTIST_PENALTY_COEFF

            final_score = SEMANTIC_WEIGHT * semantic_score + RANK_WEIGHT * rank_score - penalty

            if final_score > max_score:
                max_score = final_score
                best_track = cand_id

        return best_track if best_track is not None else candidates[0]

    def _select_anchor(self, history, prev_track):
        track_time = defaultdict(float)
        for track, listened_time in history:
            track_time[track] += listened_time

        if not track_time:
            return prev_track

        anchors = list(track_time.keys())
        weights = [track_time[t] for t in anchors]

        while anchors:
            chosen = random.choices(anchors, weights=weights, k=1)[0]
            if chosen in self.track_embeddings:
                return chosen

            idx = anchors.index(chosen)
            anchors.pop(idx)
            weights.pop(idx)

        return prev_track

    def _load_user_history(self, user):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)

        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history