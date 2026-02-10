"""Transcript parsing, segment matching, cluster mapping, and validation logic."""

from collections import Counter, defaultdict

from .config import (
    MIN_CLUSTER_VOTES,
    MIN_MATCH_RATIO,
    MULTI_SPEAKER,
    PLACEHOLDER_RE,
    SPEAKER_RE,
    ClusterMap,
    EpResult,
    TLine,
)
from .discovery import _best_score, _normalize, _word_overlap, _word_set
from .speakers import _canon


def parse_transcript(text: str) -> list[TLine]:
    lines = []
    for i, raw in enumerate(text.split("\n"), 1):
        stripped = raw.strip()
        if not stripped:
            continue
        is_scene = stripped.startswith("[")
        is_ph = False
        speaker = ""
        dialogue = stripped

        m = SPEAKER_RE.match(stripped)
        if m:
            speaker = m.group(1).strip()
            dialogue = stripped[m.end():].strip()
        else:
            pm = PLACEHOLDER_RE.match(stripped)
            if pm:
                is_ph = True
                dialogue = stripped[pm.end():].strip()

        lines.append(TLine(
            line_num=i, raw=stripped, speaker=speaker,
            text=dialogue, is_scene=is_scene, is_placeholder=is_ph,
        ))
    return lines


def match_to_segments(lines: list[TLine], segments: list[dict]) -> None:
    """Match transcript lines to whisperX segments.  Mutates lines in-place.

    Each segment already has text + speaker, so we fuzzy-match transcript text
    to segment text and read the speaker directly.  No separate diarization
    lookup needed.
    """
    seg_idx = 0
    seg_norm = [_normalize(s.get("text", "")) for s in segments]
    seg_words = [_word_set(n) for n in seg_norm]

    for tl in lines:
        if tl.is_scene or not tl.text:
            continue
        text = _normalize(tl.text)
        if not text:
            continue
        tl_words = _word_set(text)
        best_r = 0.0
        best_spk = ""
        best_s, best_e = -1.0, -1.0

        lo = max(0, seg_idx - 5)
        hi = min(len(segments), seg_idx + 25)

        for j in range(lo, hi):
            seg = segments[j]

            if _word_overlap(tl_words, seg_words[j]) > 0.15:
                r = _best_score(text, seg_norm[j])
                if r > best_r:
                    best_r = r
                    best_spk = seg.get("speaker", "")
                    best_s, best_e = seg["start"], seg["end"]

            # Concatenate adjacent segments (transcript line may span multiple)
            cat_words = set(seg_words[j])
            cat = seg_norm[j]
            cat_e = seg["end"]
            for k in range(j + 1, min(j + 4, len(segments))):
                cat += " " + seg_norm[k]
                cat_e = segments[k]["end"]
                cat_words |= seg_words[k]
                if _word_overlap(tl_words, cat_words) > 0.15:
                    cr = _best_score(text, cat)
                    if cr > best_r:
                        best_r = cr
                        best_spk = seg.get("speaker", "")
                        best_s = seg["start"]
                        best_e = cat_e

            if best_r >= 0.85:
                break

        # Wider fallback
        if best_r < MIN_MATCH_RATIO and len(text) > 10:
            for j in range(len(segments)):
                if lo <= j < hi:
                    continue
                if _word_overlap(tl_words, seg_words[j]) < 0.2:
                    continue
                r = _best_score(text, seg_norm[j])
                if r > best_r:
                    best_r = r
                    best_spk = segments[j].get("speaker", "")
                    best_s = segments[j]["start"]
                    best_e = segments[j]["end"]
                    if r >= 0.85:
                        break

        if best_r >= MIN_MATCH_RATIO:
            tl.w_start, tl.w_end, tl.w_ratio = best_s, best_e, best_r
            tl.dia_speaker = best_spk
            for j in range(seg_idx, len(segments)):
                if segments[j]["start"] >= best_e:
                    seg_idx = j
                    break


def build_cluster_map(lines: list[TLine]) -> dict[str, ClusterMap]:
    votes: dict[str, Counter] = defaultdict(Counter)
    for tl in lines:
        if not tl.speaker or not tl.dia_speaker:
            continue
        if tl.speaker.lower() in MULTI_SPEAKER:
            continue
        # Canonicalize so "Lumpy Space Princess" and "LSP" count together
        votes[tl.dia_speaker][_canon(tl.speaker)] += 1

    mapping = {}
    for cluster, ctr in votes.items():
        top, n = ctr.most_common(1)[0]
        total = sum(ctr.values())
        mapping[cluster] = ClusterMap(cluster, top, n, total)
    return mapping


def build_merged_character_map(
    cmap: dict[str, ClusterMap],
) -> dict[str, tuple[int, int, float, set[str]]]:
    """Merge clusters that mapped to the same character.

    Returns {character: (total_votes, total_lines, confidence, cluster_ids)}.
    """
    char_cms: dict[str, list[ClusterMap]] = defaultdict(list)
    for cm in cmap.values():
        char_cms[cm.character].append(cm)
    merged = {}
    for char, cms in char_cms.items():
        total_votes = sum(cm.votes for cm in cms)
        total_lines = sum(cm.total for cm in cms)
        merged[char] = (
            total_votes, total_lines,
            total_votes / max(total_lines, 1),
            {cm.cluster for cm in cms},
        )
    return merged


def _context_speaker(lines: list[TLine], idx: int, window: int = 15) -> str:
    """Return speaker if nearest labeled lines on both sides agree."""
    prev = ""
    for j in range(idx - 1, max(idx - window, -1), -1):
        if lines[j].is_scene or not lines[j].text:
            continue
        if lines[j].speaker:
            prev = lines[j].speaker
            break
    nxt = ""
    for j in range(idx + 1, min(idx + window, len(lines))):
        if lines[j].is_scene or not lines[j].text:
            continue
        if lines[j].speaker:
            nxt = lines[j].speaker
            break
    if prev and nxt and prev == nxt:
        return prev
    return ""


def _same_cluster_neighbor(lines: list[TLine], idx: int, window: int = 6) -> str:
    """Return speaker of nearest labeled line sharing the same diarization cluster."""
    tl = lines[idx]
    if not tl.dia_speaker:
        return ""
    for j in range(idx - 1, max(idx - window, -1), -1):
        if lines[j].is_scene or not lines[j].text:
            continue
        if lines[j].speaker and lines[j].dia_speaker == tl.dia_speaker:
            return lines[j].speaker
        if lines[j].speaker:
            break  # different cluster — stop
    for j in range(idx + 1, min(idx + window, len(lines))):
        if lines[j].is_scene or not lines[j].text:
            continue
        if lines[j].speaker and lines[j].dia_speaker == tl.dia_speaker:
            return lines[j].speaker
        if lines[j].speaker:
            break
    return ""


def _infer_speaker(
    lines: list[TLine], idx: int,
    cmap: dict[str, ClusterMap],
    merged: dict[str, tuple[int, int, float, set[str]]],
) -> str:
    """Infer speaker for an unlabeled line using multiple signals."""
    tl = lines[idx]

    # Cluster signal
    cluster_char = ""
    m_votes, m_conf = 0, 0.0
    if tl.dia_speaker and tl.dia_speaker in cmap:
        cluster_char = cmap[tl.dia_speaker].character
        if cluster_char in merged:
            m_votes, _, m_conf, _ = merged[cluster_char]

    # 1. Strong merged cluster alone
    if cluster_char and m_conf >= 0.5 and m_votes >= MIN_CLUSTER_VOTES:
        return cluster_char

    # Context signals
    ctx = _context_speaker(lines, idx)
    nbr = _same_cluster_neighbor(lines, idx)

    # 2. Cluster + same-cluster neighbor agree
    if cluster_char and nbr == cluster_char:
        return cluster_char

    # 3. Cluster + context agree
    if cluster_char and ctx == cluster_char:
        return cluster_char

    # 4. Same-cluster neighbor alone
    if nbr:
        return nbr

    # 5. Context alone (both labeled neighbors agree)
    if ctx:
        return ctx

    return ""


def validate_and_fix(lines: list[TLine], cmap: dict[str, ClusterMap]) -> EpResult:
    r = EpResult()
    merged = build_merged_character_map(cmap)

    # Build canon-keyed merged map for alias-aware lookups
    canon_merged: dict[str, tuple[int, int, float, set[str]]] = {}
    for char, data in merged.items():
        ckey = _canon(char)
        if ckey in canon_merged:
            prev = canon_merged[ckey]
            canon_merged[ckey] = (
                prev[0] + data[0], prev[1] + data[1],
                (prev[0] + data[0]) / max(prev[1] + data[1], 1),
                prev[3] | data[3],
            )
        else:
            canon_merged[ckey] = data

    for i, tl in enumerate(lines):
        if tl.is_scene or not tl.text:
            continue
        r.total += 1

        if tl.speaker:
            r.labeled += 1
            if tl.dia_speaker and tl.dia_speaker in cmap:
                cm = cmap[tl.dia_speaker]
                cs = _canon(tl.speaker)
                cc = _canon(cm.character)
                # Check merged: line's speaker might match a different cluster
                if cc == cs:
                    tl.validation = "agree"
                    r.agree += 1
                elif cs in canon_merged and tl.dia_speaker in canon_merged[cs][3]:
                    # Speaker is in the set of clusters for this character
                    tl.validation = "agree"
                    r.agree += 1
                else:
                    tl.validation = "disagree"
                    r.disagree += 1
                    r.disagree_details.append({
                        "line": tl.line_num,
                        "text": tl.text[:60],
                        "transcript": tl.speaker,
                        "diarization": cm.character,
                        "cluster": tl.dia_speaker,
                        "conf": round(cm.confidence, 2),
                        "w_start": tl.w_start,
                        "w_end": tl.w_end,
                    })
            else:
                tl.validation = "unmatched"
        else:
            r.unlabeled += 1
            if not tl.is_placeholder:
                continue  # continuation lines -- not fixable
            inferred = _infer_speaker(lines, i, cmap, merged)
            if inferred:
                tl.inferred = _canon(inferred)
                r.fixed += 1
            else:
                r.unknown += 1

    r.n_clusters = len(cmap)
    r.cluster_info = {
        k: {"character": v.character, "votes": v.votes, "conf": round(v.confidence, 2)}
        for k, v in cmap.items()
    }
    return r


def format_fixed(lines: list[TLine], original: str) -> str:
    fixes = {tl.line_num: tl for tl in lines if tl.inferred}
    out = []
    for i, raw in enumerate(original.split("\n"), 1):
        if i in fixes:
            tl = fixes[i]
            out.append(f"{tl.inferred}:  {tl.text}")
        else:
            out.append(raw.rstrip())
    text = "\n".join(out)
    return text if text.endswith("\n") else text + "\n"
