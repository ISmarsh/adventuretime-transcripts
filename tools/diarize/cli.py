"""CLI entry point: argparse setup and subcommand dispatch."""

import argparse
import warnings
from pathlib import Path

from .config import (
    DEFAULT_VIDEO_DIRS,
    MAX_EMBED_DURATION,
    MIN_EMBED_DURATION,
    WHISPER_MODEL,
)
from .cmd_auto_label import cmd_auto_label
from .cmd_embed_clusters import cmd_embed_clusters
from .cmd_embed_label import cmd_embed_label
from .cmd_process import cmd_process
from .cmd_status import cmd_status
from .cmd_validate import cmd_validate

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combined whisperX diarization + validation for "
                    "Adventure Time transcripts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # process subcommand
    p_proc = sub.add_parser("process", help="Run whisperX pipeline on video files")
    p_proc.add_argument(
        "--video-dirs", nargs="+", type=Path, default=DEFAULT_VIDEO_DIRS,
        help="Directories to search for video files",
    )
    p_proc.add_argument("--output-dir", default="diarization", help="Output directory")
    p_proc.add_argument("--workers", type=int, default=2, help="Parallel workers (default: 2)")
    p_proc.add_argument("--whisper-model", default=WHISPER_MODEL, help="Whisper model")
    p_proc.add_argument("--series", choices=["all", "at", "dl", "fc"], default="all")
    p_proc.add_argument("--season", type=int, help="Filter by season")
    p_proc.add_argument("--episode", nargs="+", help="Filter by episode ID(s) (e.g. AT.S09E11 AT.S08E08)")
    p_proc.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                        help="Device for inference (default: auto-detect)")
    p_proc.add_argument("--dry-run", action="store_true", help="Preview mapping only")
    p_proc.add_argument("--force", action="store_true", help="Re-process existing output")
    p_proc.add_argument(
        "--audio-track", type=int, default=None,
        help="Audio stream index to extract (e.g. 1 for second track). "
             "Default: let ffmpeg pick the default stream.",
    )

    # validate subcommand
    p_val = sub.add_parser("validate", help="Validate/fix transcript speaker labels")
    p_val.add_argument("--episode", nargs="+", help="Filter by episode ID(s) (e.g. AT.S09E11 AT.S08E08)")
    p_val.add_argument("--series", help="Filter by series: at, dl, fc")
    p_val.add_argument("--season", type=int, help="Filter by season")
    p_val.add_argument("--unlabeled-only", action="store_true", help="Only ??? episodes")
    p_val.add_argument("--write", action="store_true", help="Apply fixes to transcripts")
    p_val.add_argument("--dry-run", action="store_true", help="Preview fixes")
    p_val.add_argument("--force", action="store_true", help="Reprocess validated episodes")
    p_val.add_argument("--limit", type=int, help="Max episodes per run")
    p_val.add_argument("--diarization-dir", default="diarization", help="JSON directory")

    # embed-clusters subcommand
    p_eclust = sub.add_parser(
        "embed-clusters",
        help="Build per-cluster embeddings from anonymous speakers")
    p_eclust.add_argument(
        "--episode", nargs="+",
        help="Episode ID(s). Omit to auto-discover unprocessed episodes.")
    p_eclust.add_argument("--series", choices=["all", "at", "dl", "fc"], default="all")
    p_eclust.add_argument("--season", type=int, help="Filter by season")
    p_eclust.add_argument(
        "--video-dirs", nargs="+", type=Path, default=DEFAULT_VIDEO_DIRS,
        help="Directories to search for video files",
    )
    p_eclust.add_argument("--force", action="store_true", help="Re-process existing clusters")
    p_eclust.add_argument("--limit", type=int, help="Max episodes to process")
    p_eclust.add_argument(
        "--min-duration", type=float, default=MIN_EMBED_DURATION,
        help="Min segment duration in seconds (default: 1.5)")
    p_eclust.add_argument(
        "--max-duration", type=float, default=MAX_EMBED_DURATION,
        help="Max segment duration in seconds (default: 15)")
    p_eclust.add_argument(
        "--samples", type=int, default=5,
        help="Sample dialogue lines per cluster (default: 5)")
    p_eclust.add_argument(
        "--diarization-dir", default="diarization", help="JSON directory")
    p_eclust.add_argument(
        "--audio-track", type=int, default=None,
        help="Audio stream index to extract (e.g. 1 for second track)",
    )

    # embed-label subcommand
    p_elabel = sub.add_parser(
        "embed-label",
        help="Save cluster->character mapping and merge into profiles")
    p_elabel.add_argument(
        "--episode", required=True, help="Episode ID (e.g. AT.S08E08)")
    p_elabel.add_argument(
        "--map", nargs="+", required=True,
        help="Cluster=Character mappings (e.g. SPEAKER_00=Finn)")
    p_elabel.add_argument(
        "--skip", nargs="*", default=[],
        help="Clusters to skip (noise, music, etc.)")
    p_elabel.add_argument(
        "--replace", action="store_true",
        help="Replace existing label file instead of merging")
    p_elabel.add_argument(
        "--diarization-dir", default="diarization", help="JSON directory")

    # auto-label subcommand
    p_auto = sub.add_parser(
        "auto-label",
        help="Propose/apply cluster labels using voice profiles")
    p_auto.add_argument(
        "--episode", nargs="+",
        help="Episode ID(s). Omit to auto-discover unlabeled episodes.")
    p_auto.add_argument("--series", choices=["all", "at", "dl", "fc"], default="all")
    p_auto.add_argument("--season", type=int, help="Filter by season")
    p_auto.add_argument(
        "--threshold", type=float, default=0.60,
        help="Min cosine similarity for auto-label (default: 0.60)")
    p_auto.add_argument(
        "--char-threshold", nargs="+", metavar="CHAR=THRESH",
        help="Per-character threshold overrides (e.g. 'Ice King=0.65')")
    p_auto.add_argument(
        "--apply", action="store_true",
        help="Write label files above threshold (default: preview only)")
    p_auto.add_argument(
        "--merge", action="store_true",
        help="Merge labeled clusters into voice profiles. "
             "Use after --apply + review to avoid contamination.")
    p_auto.add_argument("--force", action="store_true",
                        help="Re-process already-labeled episodes")
    p_auto.add_argument("--additive", action="store_true",
                        help="Only process skipped clusters in existing labels "
                             "(preserves existing mappings)")
    p_auto.add_argument("--limit", type=int, help="Max episodes to process")
    p_auto.add_argument(
        "--diarization-dir", default="diarization", help="JSON directory")

    # status subcommand
    p_status = sub.add_parser("status", help="Show pipeline progress across all episodes")
    p_status.add_argument("--series", choices=["all", "at", "dl", "fc"], default="all")
    p_status.add_argument("--season", type=int, help="Filter by season")
    p_status.add_argument(
        "--video-dirs", nargs="+", type=Path, default=DEFAULT_VIDEO_DIRS,
        help="Directories to search for video files",
    )
    p_status.add_argument("--diarization-dir", default="diarization", help="JSON directory")
    p_status.add_argument(
        "--needs", choices=["process", "clusters", "labels", "any"],
        help="Show only episodes needing this stage",
    )

    args = parser.parse_args()

    if args.command == "process":
        cmd_process(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "embed-clusters":
        cmd_embed_clusters(args)
    elif args.command == "embed-label":
        cmd_embed_label(args)
    elif args.command == "auto-label":
        cmd_auto_label(args)
    elif args.command == "status":
        cmd_status(args)
