# Transcript Gap Documentation

This document tracks wiki transcript gaps that were filled with SRT conversions and the verification/correction work done on them.

## Gap Identification

The Adventure Time Wiki has incomplete transcripts for some episodes, particularly newer content. These gaps were filled by converting SRT subtitle files to wiki format.

### Fionna and Cake S01

| Episode | Title | Source | Status |
|---------|-------|--------|--------|
| E01 | Fionna Campbell | Wiki (incomplete) | Needs formatting |
| E02 | Simon Petrikov | Wiki (incomplete) | Needs formatting |
| E03 | Cake the Cat | Wiki (incomplete) | Needs formatting |
| E04 | Prismo the Wishmaster | **SRT Conversion** | Verified |
| E05 | Destiny | **SRT Conversion** | Verified |
| E06 | The Winter King | Wiki (mixed format) | Needs formatting |
| E07 | The Star | Wiki (incomplete) | Needs formatting |
| E08 | Jerry | **SRT Conversion** | Verified |
| E09 | Casper and Nova | **SRT Conversion** | Verified |
| E10 | Cheers | **SRT Conversion** | Verified |

### Fionna and Cake S02

All episodes converted from SDH subtitles and verified via video frame extraction.

| Episode | Title | Source | Status |
|---------|-------|--------|--------|
| E01 | The Hare and the Sprout | **SDH Conversion** | Verified |
| E02 | The Crocodile Who Bit a Log | **SDH Conversion** | Verified |
| E03 | The Lion of Embers | **SDH Conversion** | Verified |
| E04 | The Cat Who Tipped the Box | **SDH Conversion** | Verified |
| E05 | The Butterfly and the River | **SDH Conversion** | Verified |
| E06 | The Bird in the Clock | **SDH Conversion** | Verified |
| E07 | The Wolves Who Wandered | **SDH Conversion** | Verified |
| E08 | The Insect that Sang | **SDH Conversion** | Verified |
| E09 | The Worm and his Orchard | **SDH Conversion** | Verified |
| E10 | The Bear and the Rose | **SDH Conversion** | Verified |

### Distant Lands

All 4 episodes have wiki-scraped transcripts (E01-E04).

## SRT Conversion Issues

SRT files don't include speaker labels. When converting to wiki format, speaker attribution must be inferred from context, which introduces errors:

### Common Error Types
- **Paired character swaps** - Fionna/Cake, Simon/Betty lines get swapped
- **Scene boundary merges** - Dialogue from different scenes gets combined
- **Third-person misattribution** - "that sad man" referring to Simon attributed to Simon instead of Fionna

### Corrections Made

| Episode | Line | Original | Corrected | Evidence |
|---------|------|----------|-----------|----------|
| E04 | 61 | Simon: "I stepped through a portal in that sad man's head" | Fionna: ... | Video frame - Fionna visible, Simon not on screen |
| E04 | 186-187 | Merged Prismo lines | Split: Prismo/Scarab/Prismo | SDH timestamps show separate entries |
| E05 | 134 | Jay: "I'ma get you too, L.D.!" | Peanut: ... | Peanut addressing Little Destiny |
| E08 | 25-26 | Merged Orbo/campfire scenes | Split into separate scenes | SDH timestamps, video frame verification |
| E09 | 160-167 | Marshall/Gary dialogue swapped | Corrected attributions | Character voice patterns |
| E10 | 45-46 | Za'Baby/Scarab line split wrong | Za'Baby: "Cluck, cluck!" | Video frame shows chicken on screen |

## Verification Workflow

See [VERIFICATION-WORKFLOW.md](VERIFICATION-WORKFLOW.md) for the detailed process used to verify speaker attributions using SDH subtitles and video frame extraction.

## Video File Locations

Video files with SDH subtitle tracks can be found in:
- `D:\Shows\` - Main series, Fionna and Cake
- `S:\Shows\` - Distant Lands, additional content

## Wiki Format Requirements

For Adventure Time Wiki submission (CC-BY-SA 3.0):
- Two spaces after character colon: `Character:  dialogue`
- Scene descriptions in brackets: `[Scene description here.]`
- Inline actions: `Character:  [gasps] Dialogue here.`
