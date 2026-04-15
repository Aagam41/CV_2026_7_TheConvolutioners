"""
bot_sort_rf.py
--------------
BoT-SORT-style multi-object tracker where the FastReID appearance embedding
is replaced by a Random Forest same-identity classifier trained on SIFT
Bag-of-Visual-Words features (see rf_train.py).

Reference: https://github.com/NirAharon/BoT-SORT/blob/main/tracker/bot_sort.py
Changes vs. the reference:
  * No FastReID / deep ReID. Appearance feature = SIFT BoW histogram.
  * Appearance distance between track feat `a` and detection feat `b` is
        d(a, b) = 1 - P_same(|a - b|)
    where P_same comes from the RF classifier.
  * No Camera Motion Compensation (GMC) block; can be added independently.
"""
import pickle
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


# ============================== Kalman Filter ==============================
class KalmanFilter:
    """Constant-velocity KF on state [cx, cy, w, h, vx, vy, vw, vh]."""
    def __init__(self):
        ndim, dt = 4, 1.0
        self._F = np.eye(2 * ndim)
        for i in range(ndim):
            self._F[i, ndim + i] = dt
        self._H = np.eye(ndim, 2 * ndim)
        self._sp = 1.0 / 20.0
        self._sv = 1.0 / 160.0

    def initiate(self, m):
        mean = np.r_[m, np.zeros_like(m)]
        std = [2 * self._sp * m[2], 2 * self._sp * m[3],
               2 * self._sp * m[2], 2 * self._sp * m[3],
               10 * self._sv * m[2], 10 * self._sv * m[3],
               10 * self._sv * m[2], 10 * self._sv * m[3]]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        std_p = [self._sp * mean[2], self._sp * mean[3],
                 self._sp * mean[2], self._sp * mean[3]]
        std_v = [self._sv * mean[2], self._sv * mean[3],
                 self._sv * mean[2], self._sv * mean[3]]
        Q = np.diag(np.square(np.r_[std_p, std_v]))
        mean = self._F @ mean
        cov = self._F @ cov @ self._F.T + Q
        return mean, cov

    def update(self, mean, cov, m):
        H = self._H
        std = [self._sp * mean[2], self._sp * mean[3],
               self._sp * mean[2], self._sp * mean[3]]
        R = np.diag(np.square(std))
        S = H @ cov @ H.T + R
        K = cov @ H.T @ np.linalg.inv(S)
        y = m - H @ mean
        return mean + K @ y, cov - K @ H @ cov


# ================================ STrack ================================
class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    _kf = KalmanFilter()
    _count = 0

    @staticmethod
    def next_id():
        STrack._count += 1
        return STrack._count

    @staticmethod
    def reset_id():
        STrack._count = 0

    def __init__(self, tlwh, score, feat=None, cls=-1):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = float(score)
        self.cls = int(cls)
        self.track_id = 0
        self.state = TrackState.New
        self.is_activated = False
        self.mean = None
        self.cov = None
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0

        # Appearance
        self.smooth_feat = None
        self.curr_feat = None
        self.alpha = 0.9           # EMA weight for smooth feature
        if feat is not None:
            self.update_features(feat)

    # ---- features ----
    def update_features(self, feat):
        feat = feat.astype(np.float32)
        n = np.linalg.norm(feat)
        if n > 0:
            feat = feat / n
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat.copy()
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
            n2 = np.linalg.norm(self.smooth_feat)
            if n2 > 0:
                self.smooth_feat /= n2

    # ---- geometry helpers ----
    @staticmethod
    def tlwh_to_xywh(tlwh):
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2.0
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    # ---- Kalman ops ----
    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0
        self.mean, self.cov = STrack._kf.predict(mean_state, self.cov)

    @staticmethod
    def multi_predict(tracks):
        for t in tracks:
            if t.mean is not None:
                t.predict()

    def activate(self, frame_id):
        self.track_id = STrack.next_id()
        self.mean, self.cov = STrack._kf.initiate(self.tlwh_to_xywh(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = (frame_id == 1)
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.cov = STrack._kf.update(
            self.mean, self.cov, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = STrack.next_id()
        self.score = new_track.score
        self.cls = new_track.cls

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.cov = STrack._kf.update(
            self.mean, self.cov, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.cls = new_track.cls

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


# =============================== Distances ===============================
def ious(atlbrs, btlbrs):
    A = np.asarray(atlbrs, dtype=np.float32).reshape(-1, 4)
    B = np.asarray(btlbrs, dtype=np.float32).reshape(-1, 4)
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)), dtype=np.float32)
    out = np.zeros((len(A), len(B)), dtype=np.float32)
    area_b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    for i in range(len(A)):
        ix1 = np.maximum(A[i, 0], B[:, 0])
        iy1 = np.maximum(A[i, 1], B[:, 1])
        ix2 = np.minimum(A[i, 2], B[:, 2])
        iy2 = np.minimum(A[i, 3], B[:, 3])
        iw = np.clip(ix2 - ix1, 0, None)
        ih = np.clip(iy2 - iy1, 0, None)
        inter = iw * ih
        area_a = (A[i, 2] - A[i, 0]) * (A[i, 3] - A[i, 1])
        out[i] = inter / np.maximum(area_a + area_b - inter, 1e-6)
    return out


def iou_distance(atracks, btracks):
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    atlbrs = [t.tlbr for t in atracks]
    btlbrs = [t.tlbr for t in btracks]
    return 1.0 - ious(atlbrs, btlbrs)


# ============================ SIFT BoW features ============================
def _sift_bow(img_bgr, sift, kmeans, n_clusters):
    """Normalized SIFT BoW histogram (float32, length=n_clusters)."""
    hist = np.zeros(n_clusters, dtype=np.float32)
    if img_bgr is None or img_bgr.size == 0:
        return hist
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, desc = sift.detectAndCompute(gray, None)
    if desc is None or len(desc) == 0:
        return hist
    labels = kmeans.predict(desc.astype(np.float32))
    for lbl in labels:
        hist[lbl] += 1
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def extract_features(img, tlwh_boxes, sift, kmeans, n_clusters,
                     crop_size=(64, 64)):
    feats = []
    H, W = img.shape[:2]
    for (x, y, w, h) in tlwh_boxes:
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(W, int(x + w)), min(H, int(y + h))
        if x2 - x1 < 4 or y2 - y1 < 4:
            feats.append(np.zeros(n_clusters, dtype=np.float32))
        else:
            crop = cv2.resize(img[y1:y2, x1:x2], crop_size)
            feats.append(_sift_bow(crop, sift, kmeans, n_clusters))
    return feats


# ====================== Random Forest appearance ======================
class RFAppearance:
    """Wrap the trained RF + SIFT vocabulary and expose pairwise distance."""
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            blob = pickle.load(f)
        self.model = blob['model']
        self.kmeans = blob['kmeans']
        self.n_clusters = int(blob['n_clusters'])
        self.sift_nfeatures = int(blob.get('sift_nfeatures', 100))
        self.crop_size = tuple(blob.get('crop_size', (64, 64)))
        self.sift = cv2.SIFT_create(nfeatures=self.sift_nfeatures)

    def distance(self, track_feats, det_feats):
        """Returns NxM matrix in [0, 1]; lower = more similar."""
        N, M = len(track_feats), len(det_feats)
        if N == 0 or M == 0:
            return np.zeros((N, M), dtype=np.float32)
        tf = np.asarray(track_feats, dtype=np.float32)
        df = np.asarray(det_feats, dtype=np.float32)
        # |a - b| pair features, batched
        pairs = np.abs(tf[:, None, :] - df[None, :, :]).reshape(N * M, -1)
        # P(same_id = 1)
        classes = list(self.model.classes_)
        proba = self.model.predict_proba(pairs)
        if 1 in classes:
            p_same = proba[:, classes.index(1)]
        else:
            p_same = proba[:, -1]
        return (1.0 - p_same.reshape(N, M)).astype(np.float32)


# ============================ Assignment ============================
def linear_assignment(cost, thresh):
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    c = cost.copy()
    c[c > thresh] = thresh + 1e-4
    r, col = linear_sum_assignment(c)
    matches = []
    for i, j in zip(r, col):
        if cost[i, j] <= thresh:
            matches.append((i, j))
    mr = {i for i, _ in matches}
    mc = {j for _, j in matches}
    un_a = [i for i in range(cost.shape[0]) if i not in mr]
    un_b = [j for j in range(cost.shape[1]) if j not in mc]
    return matches, un_a, un_b


def joint_stracks(tA, tB):
    seen = {t.track_id for t in tA}
    res = list(tA)
    for t in tB:
        if t.track_id not in seen:
            res.append(t)
            seen.add(t.track_id)
    return res


def sub_stracks(tA, tB):
    ids = {t.track_id for t in tB}
    return [t for t in tA if t.track_id not in ids]


def remove_duplicate_stracks(tA, tB):
    d = iou_distance(tA, tB)
    pairs = np.where(d < 0.15)
    dupA, dupB = set(), set()
    for p, q in zip(*pairs):
        ageA = tA[p].frame_id - tA[p].start_frame
        ageB = tB[q].frame_id - tB[q].start_frame
        if ageA > ageB:
            dupB.add(q)
        else:
            dupA.add(p)
    return ([t for i, t in enumerate(tA) if i not in dupA],
            [t for i, t in enumerate(tB) if i not in dupB])


# =============================== Tracker ===============================
class BoTSORT_RF:
    def __init__(self,
                 rf_model_path,
                 frame_rate=30,
                 track_high_thresh=0.5,
                 track_low_thresh=0.1,
                 new_track_thresh=0.6,
                 match_thresh=0.8,
                 appearance_thresh=0.25,
                 proximity_thresh=0.5,
                 track_buffer=30):
        self.tracked = []
        self.lost = []
        self.removed = []
        self.frame_id = 0

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.appearance_thresh = appearance_thresh
        self.proximity_thresh = proximity_thresh
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)

        self.appearance = RFAppearance(rf_model_path)
        STrack.reset_id()

    # --------- main update ----------
    def update(self, dets, img):
        """
        dets : ndarray (N, 5) or (N, 6) in [x1,y1,x2,y2,score(,cls)]
        img  : BGR frame
        returns list[STrack] of currently active tracks
        """
        self.frame_id += 1
        activated, refound, lost_new, removed_new = [], [], [], []

        if dets is None or len(dets) == 0:
            dets = np.empty((0, 6), dtype=np.float32)
        if dets.shape[1] == 5:
            dets = np.c_[dets, -np.ones(len(dets))]

        scores = dets[:, 4]
        bboxes = dets[:, :4]           # x1y1x2y2
        classes = dets[:, 5]

        hi = scores > self.track_high_thresh
        lo = (scores > self.track_low_thresh) & (scores <= self.track_high_thresh)

        d_hi = bboxes[hi]; s_hi = scores[hi]; c_hi = classes[hi]
        d_lo = bboxes[lo]; s_lo = scores[lo]; c_lo = classes[lo]

        # Convert to tlwh for feature crop + track construction
        def to_tlwh(b):
            if len(b) == 0:
                return np.empty((0, 4), dtype=np.float32)
            return np.c_[b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]]

        tlwh_hi = to_tlwh(d_hi)
        tlwh_lo = to_tlwh(d_lo)

        feats_hi = extract_features(img, tlwh_hi,
                                    sift=self.appearance.sift,
                                    kmeans=self.appearance.kmeans,
                                    n_clusters=self.appearance.n_clusters,
                                    crop_size=self.appearance.crop_size)

        detections = [STrack(tlwh, s, f, c)
                      for tlwh, s, f, c in zip(tlwh_hi, s_hi, feats_hi, c_hi)]
        detections_low = [STrack(tlwh, s, None, c)
                          for tlwh, s, c in zip(tlwh_lo, s_lo, c_lo)]

        # Pools
        unconfirmed = [t for t in self.tracked if not t.is_activated]
        tracked = [t for t in self.tracked if t.is_activated]
        strack_pool = joint_stracks(tracked, self.lost)

        # Predict
        STrack.multi_predict(strack_pool)

        # ---------- First association: IoU + RF appearance ----------
        iou_d = iou_distance(strack_pool, detections)
        if len(strack_pool) and len(detections):
            feat_dim = detections[0].curr_feat.shape[0]
            t_feats = [t.smooth_feat if t.smooth_feat is not None
                       else np.zeros(feat_dim, dtype=np.float32)
                       for t in strack_pool]
            d_feats = [d.curr_feat for d in detections]
            app_d = self.appearance.distance(t_feats, d_feats)
            # Gates
            app_d[iou_d > self.proximity_thresh] = 1.0
            app_d[app_d > self.appearance_thresh] = 1.0
            dists = np.minimum(iou_d, app_d)
        else:
            dists = iou_d

        matches, u_track, u_det = linear_assignment(dists, thresh=self.match_thresh)
        for it, idet in matches:
            trk = strack_pool[it]
            det = detections[idet]
            if trk.state == TrackState.Tracked:
                trk.update(det, self.frame_id)
                activated.append(trk)
            else:
                trk.re_activate(det, self.frame_id, new_id=False)
                refound.append(trk)

        # ---------- Second association: low-score dets vs. remaining tracked ----------
        r_tracked = [strack_pool[i] for i in u_track
                     if strack_pool[i].state == TrackState.Tracked]
        dists2 = iou_distance(r_tracked, detections_low)
        matches, u_track2, _ = linear_assignment(dists2, thresh=0.5)
        for it, idet in matches:
            trk = r_tracked[it]
            det = detections_low[idet]
            trk.update(det, self.frame_id)
            activated.append(trk)
        for i in u_track2:
            trk = r_tracked[i]
            if trk.state != TrackState.Lost:
                trk.mark_lost()
                lost_new.append(trk)

        # ---------- Unconfirmed tracks vs. remaining high-conf detections ----------
        remaining_dets = [detections[i] for i in u_det]
        dists3 = iou_distance(unconfirmed, remaining_dets)
        matches, u_unc, u_det2 = linear_assignment(dists3, thresh=0.7)
        for iu, idet in matches:
            unconfirmed[iu].update(remaining_dets[idet], self.frame_id)
            activated.append(unconfirmed[iu])
        for i in u_unc:
            trk = unconfirmed[i]
            trk.mark_removed()
            removed_new.append(trk)

        # ---------- Init new tracks from leftover high-conf dets ----------
        for i in u_det2:
            trk = remaining_dets[i]
            if trk.score < self.new_track_thresh:
                continue
            trk.activate(self.frame_id)
            activated.append(trk)

        # ---------- Retire old lost tracks ----------
        for t in self.lost:
            if self.frame_id - t.frame_id > self.max_time_lost:
                t.mark_removed()
                removed_new.append(t)

        # ---------- Bookkeeping ----------
        self.tracked = [t for t in self.tracked if t.state == TrackState.Tracked]
        self.tracked = joint_stracks(self.tracked, activated)
        self.tracked = joint_stracks(self.tracked, refound)
        self.lost = sub_stracks(self.lost, self.tracked)
        self.lost.extend(lost_new)
        self.lost = sub_stracks(self.lost, self.removed)
        self.removed.extend(removed_new)
        self.tracked, self.lost = remove_duplicate_stracks(self.tracked, self.lost)

        return [t for t in self.tracked if t.is_activated]
