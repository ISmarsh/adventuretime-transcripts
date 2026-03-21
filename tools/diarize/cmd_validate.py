"""Validate subcommand: compare diarization against transcripts."""

import argparse
import io
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .config import (
    EPISODE_RE,
    PROGRESS_FILE,
    PROJECT_ROOT,
    TRANSCRIPT_SERIES_DIRS,
    ClusterMap,
    EpResult,
)
from .transcript import (
    build_cluster_map,
    format_fixed,
    match_to_segments,
    parse_transcript,
    validate_and_fix,
)

# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


def load_progress(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"processed": {}, "started": datetime.now().isoformat()}


def save_progress(path: Path, data: dict) -> None:
    data["updated"] = datetime.now().isoformat()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Episode discovery for validation
# ---------------------------------------------------------------------------


def discover_for_validate(root: Path, dia_dir: Path) -> list[dict]:
    """Find episodes with whisperX output, matched to transcripts."""
    eps = []
    for jp in sorted(dia_dir.glob("*.json")):
        if jp.name == PROGRESS_FILE:
            continue

        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        # Skip error-only files
        if "error" in data and "segments" not in data:
            continue

        eid = data.get("episode_id", jp.stem)

        # Find transcript
        tp = data.get("transcript_path", "")
        transcript = Path(tp) if tp and Path(tp).exists() else None

        if not transcript:
            m = EPISODE_RE.search(eid.replace(".", ""))
            if m:
                s, e = int(m.group(1)), int(m.group(2))
                code = eid.split(".")[0] if "." in eid else "AT"
                sdir = TRANSCRIPT_SERIES_DIRS.get(code, "Adventure Time")
                season_dir = root / sdir / f"Season {s:02d}"
                if season_dir.exists():
                    for txt in season_dir.glob(f"*S{s:02d}E{e:02d}*"):
                        transcript = txt
                        break

        if not transcript:
            continue

        m = EPISODE_RE.search(eid)
        season = int(m.group(1)) if m else 0

        eps.append({
            "episode_id": eid,
            "title": data.get("title", ""),
            "season": season,
            "dia_json": jp,
            "transcript": transcript,
            "segments": data.get("segments", []),
        })

    return eps


# ---------------------------------------------------------------------------
# Episode validation
# ---------------------------------------------------------------------------


def validate_episode(
    ep: dict, label_dir: Path | None = None,
) -> tuple[EpResult, list, str]:
    """Validate a single episode using whisperX segments.

    If a label file exists (from embed-label), use it as the cluster->character
    mapping instead of deriving one from transcript majority voting.
    """
    eid = ep["episode_id"]
    original = ep["transcript"].read_text(encoding="utf-8")
    lines = parse_transcript(original)

    segments = ep["segments"]
    if not segments:
        r = EpResult(episode_id=eid, title=ep["title"], error="no segments")
        return r, lines, original

    # Match transcript lines to whisperX segments (text + speaker directly)
    match_to_segments(lines, segments)

    # Check for label file (independent cluster->character mapping)
    cmap = None
    if label_dir:
        label_path = label_dir / f"{eid}.json"
        if label_path.exists():
            label_data = json.loads(label_path.read_text(encoding="utf-8"))
            speaker_map = label_data.get("speaker_map", {})
            if speaker_map:
                # Build ClusterMap from the label file
                cmap = {}
                for cluster, character in speaker_map.items():
                    total = sum(
                        1 for tl in lines if tl.dia_speaker == cluster)
                    cmap[cluster] = ClusterMap(
                        cluster=cluster,
                        character=character,
                        votes=total,
                        total=total,
                    )

    # Fall back to transcript-derived voting
    if cmap is None:
        cmap = build_cluster_map(lines)

    result = validate_and_fix(lines, cmap)
    result.episode_id = eid
    result.title = ep["title"]
    result.matched = sum(1 for tl in lines if tl.w_start >= 0 and not tl.is_scene)

    return result, lines, original


# ---------------------------------------------------------------------------
# CLI: validate subcommand
# ---------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> None:
    # Force UTF-8 output on Windows
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if sys.stderr.encoding != "utf-8":
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    dia_dir = PROJECT_ROOT / args.diarization_dir

    if not dia_dir.exists():
        print(f"Error: {dia_dir} not found", file=sys.stderr)
        sys.exit(1)

    episodes = discover_for_validate(PROJECT_ROOT, dia_dir)
    print(f"Found {len(episodes)} episodes with diarization output")

    # Apply filters
    if args.episode:
        episodes = [e for e in episodes if e["episode_id"] in args.episode]
    if args.series:
        prefix = args.series.upper() + "."
        episodes = [e for e in episodes if e["episode_id"].startswith(prefix)]
    if args.season is not None:
        episodes = [e for e in episodes if e["season"] == args.season]
    if args.unlabeled_only:
        filtered = []
        for ep in episodes:
            text = ep["transcript"].read_text(encoding="utf-8")
            if "???:" in text or "[?]:" in text:
                filtered.append(ep)
        episodes = filtered

    # Resume: skip already-processed
    progress_path = dia_dir / PROGRESS_FILE
    progress = load_progress(progress_path)
    if not args.force:
        before = len(episodes)
        episodes = [e for e in episodes if e["episode_id"] not in progress["processed"]]
        skipped = before - len(episodes)
        if skipped:
            print(f"Skipping {skipped} already-processed (use --force to redo)")

    if not episodes:
        print("Nothing to process.")
        return

    if args.limit:
        episodes = episodes[:args.limit]

    # Check for label directory (from embed-label)
    label_dir = dia_dir / "labels"
    if not label_dir.exists():
        label_dir = None

    print(f"Processing {len(episodes)} episodes\n")

    total_agree = total_disagree = total_uncertain = total_fixed = total_unknown = 0
    errors = []

    for i, ep in enumerate(episodes, 1):
        eid = ep["episode_id"]
        print(f"[{i:03d}/{len(episodes)}] {eid} {ep['title']}")

        try:
            result, lines, original = validate_episode(ep, label_dir)

            if result.error:
                print(f"  SKIP: {result.error}")
                errors.append(eid)
                progress["processed"][eid] = {
                    "ts": datetime.now().isoformat(), "error": result.error,
                }
                save_progress(progress_path, progress)
                continue

            m_pct = f"{result.matched}/{result.total}" if result.total else "0/0"
            print(f"  {result.total} dialogue ({result.labeled} labeled, "
                  f"{result.unlabeled} unlabeled) | matched {m_pct}")

            if result.cluster_info:
                parts = []
                for cid, info in sorted(result.cluster_info.items()):
                    parts.append(
                        f"{cid}={info['character']}({info['votes']},{info['conf']:.0%})"
                    )
                print(f"  Clusters: {', '.join(parts)}")
                # Show merged character view (combine split clusters)
                char_agg: dict[str, list] = defaultdict(list)
                for cid, info in result.cluster_info.items():
                    char_agg[info["character"]].append(
                        (cid, info["votes"], info["conf"]))
                multi = {c: v for c, v in char_agg.items() if len(v) > 1}
                if multi:
                    mparts = []
                    for char, clusters in sorted(multi.items()):
                        total_v = sum(v for _, v, _ in clusters)
                        cids = "+".join(c for c, _, _ in clusters)
                        mparts.append(f"{char}({total_v} votes via {cids})")
                    print(f"  Merged: {', '.join(mparts)}")

            unc_str = f", {result.uncertain} uncertain" if result.uncertain else ""
            print(f"  Validation: {result.agree} agree, {result.disagree} disagree{unc_str}"
                  + (f" | Fixed: {result.fixed}, unknown: {result.unknown}"
                     if result.unlabeled else ""))

            real_disagrees = [d for d in result.disagree_details if not d.get("uncertain")]
            uncertains = [d for d in result.disagree_details if d.get("uncertain")]
            for d in real_disagrees[:5]:
                print(f"    DISAGREE L{d['line']}: "
                      f"{d['transcript']} -> {d['diarization']} "
                      f"(cluster {d['cluster']}, conf {d['conf']:.0%})")
            if len(real_disagrees) > 5:
                print(f"    ... and {len(real_disagrees) - 5} more")
            for d in uncertains[:3]:
                print(f"    UNCERTAIN L{d['line']}: "
                      f"{d['transcript']} -> {d['diarization']} "
                      f"(cluster {d['cluster']})")
            if len(uncertains) > 3:
                print(f"    ... and {len(uncertains) - 3} more uncertain")

            # Rescore: segment-level profile comparison
            save_disagrees = result.disagree_details
            rescore_summary = None
            if getattr(args, "rescore", False) and result.disagree_details:
                from .rescore import rescore_disagrees, split_clusters

                # Cluster split analysis — get label map
                label_path = (label_dir / f"{eid}.json") if label_dir else None
                if label_path and label_path.exists():
                    ldata = json.loads(label_path.read_text(encoding="utf-8"))
                    lmap = ldata.get("speaker_map", {})
                elif result.cluster_info:
                    lmap = {k: v["character"]
                            for k, v in result.cluster_info.items()}
                else:
                    lmap = {}

                if lmap:
                    splits = split_clusters(
                        eid, ep["segments"], dia_dir, lmap)
                    if splits:
                        print("  Cluster splits:")
                        for cid, info in sorted(splits.items()):
                            parts = [f"{k}: {v}" for k, v
                                     in info["splits"].items()]
                            print(f"    {cid} ({info['assigned']}): "
                                  f"{', '.join(parts)} "
                                  f"[{info['mixed_pct']}% mixed]")

                # Per-segment rescoring
                rescored = rescore_disagrees(
                    eid, result.disagree_details, ep["segments"], dia_dir)
                save_disagrees = rescored
                n_seg = sum(1 for r in rescored
                            if r.get("rescore_type") == "segment")
                n_cent = sum(1 for r in rescored
                             if r.get("rescore_type") == "centroid")
                n_t = sum(1 for r in rescored
                          if r.get("rescore_verdict") == "transcript")
                n_d = sum(1 for r in rescored
                          if r.get("rescore_verdict") == "diarization")
                n_scored = n_t + n_d
                rescore_summary = {
                    "segment": n_seg, "centroid": n_cent,
                    "transcript": n_t, "diarization": n_d,
                    "no_profile": len(rescored) - n_scored,
                }

                if n_scored > 0:
                    print(f"  Rescore ({n_seg} segment, {n_cent} centroid):")
                    for r in rescored[:5]:
                        v = r.get("rescore_verdict", "")
                        if not v:
                            continue
                        ts = r.get("rescore_transcript", 0)
                        ds = r.get("rescore_diarization", 0)
                        mark = "T" if v == "transcript" else "D"
                        print(f"    L{r['line']}: {r['transcript']}({ts:.2f})"
                              f" vs {r['diarization']}({ds:.2f})"
                              f" -> {mark}")
                    if n_scored > 5:
                        print(f"    ... and {n_scored - 5} more")
                    print(f"  Rescore summary: {n_t} transcript, "
                          f"{n_d} diarization "
                          f"({n_scored} rescored, "
                          f"{len(rescored) - n_scored} no profile)")

            fix_lines = [tl for tl in lines if tl.inferred]
            if args.write and fix_lines:
                new_text = format_fixed(lines, original)
                ep["transcript"].write_text(new_text, encoding="utf-8")
                print(f"  WROTE {len(fix_lines)} fixes to {ep['transcript'].name}")
            elif args.dry_run and fix_lines:
                for tl in fix_lines[:10]:
                    print(f"    FIX L{tl.line_num}: -> {tl.inferred}:  {tl.text[:50]}")
                if len(fix_lines) > 10:
                    print(f"    ... and {len(fix_lines) - 10} more fixes")

            total_agree += result.agree
            total_disagree += result.disagree
            total_uncertain += result.uncertain
            total_fixed += result.fixed
            total_unknown += result.unknown

            ep_progress = {
                "ts": datetime.now().isoformat(),
                "total": result.total,
                "labeled": result.labeled,
                "unlabeled": result.unlabeled,
                "matched": result.matched,
                "agree": result.agree,
                "disagree": result.disagree,
                "uncertain": result.uncertain,
                "fixed": result.fixed,
                "unknown": result.unknown,
                "clusters": result.cluster_info,
                "disagreements": save_disagrees,
            }
            if rescore_summary:
                ep_progress["rescore"] = rescore_summary
            progress["processed"][eid] = ep_progress
            save_progress(progress_path, progress)

        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            errors.append(eid)
            progress["processed"][eid] = {
                "ts": datetime.now().isoformat(), "error": str(e),
            }
            save_progress(progress_path, progress)

    total_processed = len(episodes) - len(errors)
    print(f"\n{'=' * 60}")
    print(f"Processed: {total_processed}/{len(episodes)}"
          + (f" ({len(errors)} errors)" if errors else ""))
    unc_total = f", {total_uncertain} uncertain" if total_uncertain else ""
    print(f"Validation: {total_agree} agree, {total_disagree} disagree{unc_total}")
    if total_fixed or total_unknown:
        print(f"Fixes: {total_fixed} applied, {total_unknown} still unknown")
    if errors:
        print(f"Failed: {', '.join(errors)}")
