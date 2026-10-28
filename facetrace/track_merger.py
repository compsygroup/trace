"""Merge tracks belonging to the same identity, split by long gaps.

Two tracks merge when their AdaFace embeddings are similar enough and they pass a
temporal-overlap gate (can't coexist in time) and a geometric-plausibility gate.
"""
import statistics
from collections import defaultdict

import numpy as np

_FRAME_KEYS = ["frames", "times", "areas", "det_conf", "bboxes", "landmarks", "sub_track_ids"]


def _bbox_center_size(bbox):
    if not bbox or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = bbox[:4]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, max(x2 - x1, y2 - y1))


def _track_geom(td):
    times = td.get("times", [])
    bboxes = td.get("bboxes", [])
    if not times or not bboxes:
        return None
    order = sorted(range(len(times)), key=lambda i: times[i])
    first_i, last_i = order[0], order[-1]
    start_bb = bboxes[first_i] if first_i < len(bboxes) else []
    end_bb = bboxes[last_i] if last_i < len(bboxes) else []
    tail = order[-5:] if len(order) >= 2 else order
    centers = [(_bbox_center_size(bboxes[i]), times[i]) for i in tail
               if i < len(bboxes) and _bbox_center_size(bboxes[i]) is not None]
    vx = vy = 0.0
    if len(centers) >= 2:
        (c0, t0), (c1, t1) = centers[0], centers[-1]
        dt = max(t1 - t0, 1e-3)
        vx = (c1[0] - c0[0]) / dt
        vy = (c1[1] - c0[1]) / dt
    return dict(
        start_t=times[first_i], end_t=times[last_i],
        start_c=_bbox_center_size(start_bb), end_c=_bbox_center_size(end_bb),
        vx=vx, vy=vy,
    )


def _temporally_overlaps(ga, gb, tol=0.3):
    return min(ga["end_t"], gb["end_t"]) - max(ga["start_t"], gb["start_t"]) > tol


def _geometrically_plausible(ga, gb, max_gap_sec=3.0):
    if ga["end_t"] <= gb["start_t"]:
        first, second = ga, gb
    elif gb["end_t"] <= ga["start_t"]:
        first, second = gb, ga
    else:
        return True

    gap = second["start_t"] - first["end_t"]
    if gap > max_gap_sec:
        return True

    e, s = first["end_c"], second["start_c"]
    if e is None or s is None:
        return True

    pred_x = e[0] + first["vx"] * gap
    pred_y = e[1] + first["vy"] * gap
    dist = ((pred_x - s[0]) ** 2 + (pred_y - s[1]) ** 2) ** 0.5
    face = max(e[2], s[2], 30.0)
    budget = face * (2.0 + 3.0 * gap)
    return dist <= budget


def _normalize(vec):
    e = np.asarray(vec, dtype=np.float32)
    return e / (np.linalg.norm(e) + 1e-8)


def _top_embs_normalized(td):
    top = td.get("top_embs")
    if top:
        return [_normalize(v) for (_, v) in top]
    if td.get("best_emb") is not None:
        return [_normalize(td["best_emb"])]
    return []


def _max_pairwise_sim(embs_a, embs_b):
    best = -1.0
    for ea in embs_a:
        for eb in embs_b:
            s = float(np.dot(ea, eb))
            if s > best:
                best = s
    return best


def _finalize_single(td, tid):
    td = dict(td)
    td.pop("best_emb", None)
    td.pop("top_embs", None)
    td.setdefault("merged_from", [tid])
    td.setdefault("sub_track_ids", [str(tid)] * len(td.get("times", [])))
    td["n_frames"] = len(td.get("times", []))
    td["median_area"] = int(statistics.median(td["areas"])) if td.get("areas") else 0
    return td


def merge_same_person_tracks(track_data: dict, threshold: float = 0.40) -> dict:
    tids = list(track_data.keys())
    if len(tids) <= 1:
        return {t: _finalize_single(track_data[t], t) for t in tids}

    emb_tids = [t for t in tids if _top_embs_normalized(track_data[t])]
    if len(emb_tids) <= 1:
        return {t: _finalize_single(track_data[t], t) for t in tids}

    embs = {t: _top_embs_normalized(track_data[t]) for t in emb_tids}
    geom = {t: _track_geom(track_data[t]) for t in emb_tids}

    parent = {t: t for t in emb_tids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i, ti in enumerate(emb_tids):
        for tj in emb_tids[i + 1:]:
            ga, gb = geom[ti], geom[tj]
            if ga is not None and gb is not None:
                if _temporally_overlaps(ga, gb):
                    continue
                if not _geometrically_plausible(ga, gb):
                    continue
            if _max_pairwise_sim(embs[ti], embs[tj]) >= threshold:
                union(ti, tj)

    groups: dict = defaultdict(list)
    for t in emb_tids:
        groups[find(t)].append(t)

    new_data: dict = {}
    for group in groups.values():
        new_tid = min(group)
        if len(group) == 1:
            new_data[new_tid] = _finalize_single(track_data[new_tid], new_tid)
            continue

        group_s = sorted(group, key=lambda t: track_data[t]["times"][0]
                         if track_data[t].get("times") else 0)
        buckets = {k: [] for k in _FRAME_KEYS}
        best_crop_b64, best_area = "", 0
        sub_crops: dict = {}
        for t in group_s:
            td = track_data[t]
            nf = len(td.get("times", []))
            for k in _FRAME_KEYS:
                default = [str(t)] * nf if k == "sub_track_ids" else [None] * nf
                buckets[k].extend(td.get(k, default))
            if td.get("best_crop_b64"):
                sub_crops[str(t)] = td["best_crop_b64"]
            mx = max(td["areas"]) if td.get("areas") else 0
            if mx > best_area:
                best_area, best_crop_b64 = mx, td.get("best_crop_b64", "")

        order = np.argsort(buckets["times"], kind="stable")
        merged = {k: [buckets[k][i] for i in order] for k in _FRAME_KEYS}
        merged.update(
            best_crop_b64=best_crop_b64, best_area=best_area,
            n_frames=len(buckets["times"]),
            median_area=int(statistics.median(buckets["areas"])) if buckets["areas"] else 0,
            merged_from=sorted(group), sub_crops=sub_crops,
        )
        new_data[new_tid] = merged

    for t in tids:
        if t not in emb_tids and t not in new_data:
            new_data[t] = _finalize_single(track_data[t], t)

    return new_data
