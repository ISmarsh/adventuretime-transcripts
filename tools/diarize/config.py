"""Constants, configuration, and data classes for the diarization pipeline."""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

EPISODE_RE = re.compile(r"S(\d{2})E(\d{2,3})")
MULTI_EP_RE = re.compile(r"S(\d{2})E(\d{2,3})-E?(\d{2,3})")
SPEAKER_RE = re.compile(r"^([A-Z][A-Za-z \u2019'.\-()]+):\s")
PLACEHOLDER_RE = re.compile(r"^\?\?\?:\s+|^([A-Z][A-Za-z .\-]+) \[\?\]:\s+")
_NORM_RE = re.compile(r"[^\w\s']")
_SPACE_RE = re.compile(r"\s+")

FC_PATTERN = re.compile(
    r"Fionna[. ]and[. ]Cake[. ]S\d|Fionna and Cake[/\\]Season"
)

# ---------------------------------------------------------------------------
# Model / path configuration
# ---------------------------------------------------------------------------

PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
WHISPER_MODEL = "medium.en"
if sys.platform == "win32":
    DEFAULT_VIDEO_DIRS = [Path("D:/Shows"), Path("S:/Shows")]
elif Path("/shows-d").is_dir():
    # Docker container with mounted volumes
    DEFAULT_VIDEO_DIRS = [Path("/shows-d"), Path("/shows-s")]
else:
    # WSL2 with Windows drives mounted
    DEFAULT_VIDEO_DIRS = [Path("/mnt/d/Shows"), Path("/mnt/s/Shows")]

TRANSCRIPT_SERIES_DIRS = {
    "AT": "Adventure Time",
    "DL": "Adventure Time Distant Lands",
    "FC": "Adventure Time Fionna and Cake",
}

# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------

MIN_MATCH_RATIO = 0.55
MIN_FIX_CONFIDENCE = 0.6
MIN_CLUSTER_VOTES = 3
MULTI_SPEAKER = frozenset({"both", "all", "everyone", "together"})

# ---------------------------------------------------------------------------
# Embedding constants
# ---------------------------------------------------------------------------

ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_SAVEDIR = "diarization/models/ecapa"
EMBED_DIM = 192
MIN_EMBED_DURATION = 1.5
MAX_EMBED_DURATION = 15.0
SEASON_BUCKETS: dict[str, list[tuple[int, int]]] = {
    "Finn": [(1, 3), (4, 6), (7, 10)],
}
BUCKET_RE = re.compile(r"(.+)_S(\d{2})-S(\d{2})$")

# Character alias resolution (from the-enchiridion sibling project)
# 4 parents: config.py -> diarize/ -> tools/ -> project root -> workspace root
ENCHIRIDION_CHARACTERS = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "the-enchiridion" / "src" / "data" / "characters.json"
)

# Preferred canonical transcript name per character id
CANONICAL_SPEAKER: dict[str, str] = {
    "finn": "Finn",
    "jake": "Jake",
    "bmo": "BMO",
    "princess-bubblegum": "Princess Bubblegum",
    "marceline": "Marceline",
    "ice-king": "Ice King",
    "lumpy-space-princess": "LSP",
    "lady-rainicorn": "Lady Rainicorn",
    "flame-princess": "Flame Princess",
    "hunson-abadeer": "Hunson Abadeer",
    "patience-st-pim": "Patience St. Pim",
    "peppermint-butler": "Peppermint Butler",
    "lemongrab": "Lemongrab",
    "betty-grof": "Betty",
    "gunter": "Gunter",
    "fern": "Fern",
    "neptr": "NEPTR",
    "susan-strong": "Susan Strong",
    "magic-man": "Magic Man",
    "tree-trunks": "Tree Trunks",
    "cinnamon-bun": "Cinnamon Bun",
    "banana-man": "Banana Man",
    "martin-mertens": "Martin",
    "fionna": "Fionna",
    "cake": "Cake",
    "prince-gumball": "Prince Gumball",
    "marshall-lee": "Marshall Lee",
}

# Aliases that should NOT be merged (different voice / performance)
VOICE_SEPARATE: frozenset[str] = frozenset({
    "Simon", "Simon Petrikov",          # Tom Kenny normal voice ≠ Ice King
    "Normal Man", "King Man",           # Magic Man reformed
    "Dirt Beer Guy",                    # Different from Root Beer Guy
    "Fern", "Grass Finn", "Fern the Human",  # Different from Finn
    "Ice Thing",                        # Different from Gunter
    "Sweet P", "Sweet Pig-Trunks",      # Different from The Lich
    "Punch Bowl",                       # Different from Uncle Gumbald
    "Manfried",                         # Different from Aunt Lolly
    "Crunchy",                          # Different from Cousin Chicle
    "Nectr",                            # Different from Lemongrab
    "Winter King",                      # Different from Ice King (F&C)
})

# Manual transcript abbreviations not in characters.json aliases
MANUAL_ALIASES: dict[str, str] = {
    "Patience": "Patience St. Pim",
    "Hunson": "Hunson Abadeer",
    "Pep But": "Peppermint Butler",
    "Pep-But": "Peppermint Butler",
    "Simon Petrikov": "Simon",  # Same voice, separate from Ice King
    "Red-tie businessman": "Business Men",
    "Businessmen": "Business Men",
    "Imaginary Neptr": "NEPTR",
}

PROGRESS_FILE = "validation_progress.json"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    series: str
    season: int
    episode: int
    title: str
    transcript_path: Path
    video_path: Path | None = None

    @property
    def episode_id(self) -> str:
        return f"{self.series}.S{self.season:02d}E{self.episode:02d}"

    @property
    def output_filename(self) -> str:
        return f"{self.episode_id}.json"


@dataclass
class TLine:
    line_num: int
    raw: str
    speaker: str
    text: str
    is_scene: bool
    is_placeholder: bool
    w_start: float = -1.0
    w_end: float = -1.0
    w_ratio: float = 0.0
    dia_speaker: str = ""
    inferred: str = ""
    validation: str = ""


@dataclass
class ClusterMap:
    cluster: str
    character: str
    votes: int
    total: int

    @property
    def confidence(self) -> float:
        return self.votes / max(self.total, 1)


@dataclass
class EpResult:
    episode_id: str = ""
    title: str = ""
    total: int = 0
    labeled: int = 0
    unlabeled: int = 0
    matched: int = 0
    n_clusters: int = 0
    agree: int = 0
    disagree: int = 0
    fixed: int = 0
    unknown: int = 0
    disagree_details: list = field(default_factory=list)
    cluster_info: dict = field(default_factory=dict)
    error: str = ""
