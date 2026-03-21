#!/usr/bin/env python3
"""Convert PGS/SUP bitmap subtitles to SRT using Tesseract OCR.

Requirements:
    pip install pillow pytesseract
    Tesseract OCR must be installed: https://github.com/tesseract-ocr/tesseract
    Windows: winget install UB-Mannheim.TesseractOCR
"""

import argparse
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytesseract
from PIL import Image

# Configure Tesseract: check PATH first, fall back to Windows default
_tesseract_path = shutil.which('tesseract')
if not _tesseract_path and sys.platform == 'win32':
    _tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if _tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_path

# Default duration when subtitle end time can't be determined from PGS data
DEFAULT_SUBTITLE_DURATION_SECS = 3


@dataclass
class Subtitle:
    start_time: float
    end_time: float
    text: str


def format_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_pgs(data: bytes):
    """Parse PGS/SUP file and extract subtitle images with timing."""
    pos = 0
    palette = {}
    current_image = None
    current_pts = 0
    subtitles = []
    pending_sub = None

    while pos < len(data) - 13:
        # Check for PG magic
        if data[pos:pos+2] != b'PG':
            pos += 1
            continue

        pts = struct.unpack('>I', data[pos+2:pos+6])[0] / 90000.0
        seg_type = data[pos+10]
        seg_size = struct.unpack('>H', data[pos+11:pos+13])[0]
        segment = data[pos+13:pos+13+seg_size]
        pos += 13 + seg_size

        if seg_type == 0x14:  # PDS - Palette Definition
            palette = parse_palette(segment)
        elif seg_type == 0x15:  # ODS - Object Definition
            img_data = parse_ods(segment)
            if img_data:
                current_image = decode_image(img_data, palette)
                current_pts = pts
        elif seg_type == 0x16:  # PCS - Presentation Composition
            num_objects = segment[7] if len(segment) > 7 else 0

            # If we have a pending subtitle and this PCS has no objects, it's the end
            if pending_sub and num_objects == 0:
                pending_sub.end_time = pts
                subtitles.append(pending_sub)
                pending_sub = None

            # If we have an image ready, OCR it
            if current_image and num_objects > 0:
                text = ocr_image(current_image)
                if text.strip():
                    pending_sub = Subtitle(current_pts, pts + DEFAULT_SUBTITLE_DURATION_SECS, text)
                current_image = None
        elif seg_type == 0x80:  # END segment
            pass

    # Add any remaining subtitle
    if pending_sub:
        subtitles.append(pending_sub)

    return subtitles


def parse_palette(data: bytes) -> dict:
    """Parse palette definition segment."""
    if len(data) < 2:
        return {}

    palette = {}
    pos = 2  # Skip palette ID and version
    while pos + 5 <= len(data):
        idx = data[pos]
        y = data[pos + 1]
        cr = data[pos + 2]
        cb = data[pos + 3]
        alpha = data[pos + 4]

        # YCrCb to RGB conversion
        r = max(0, min(255, int(y + 1.402 * (cr - 128))))
        g = max(0, min(255, int(y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128))))
        b = max(0, min(255, int(y + 1.772 * (cb - 128))))

        palette[idx] = (r, g, b, alpha)
        pos += 5

    return palette


def parse_ods(data: bytes):
    """Parse Object Definition Segment."""
    if len(data) < 11:
        return None

    flags = data[3]
    if flags & 0x80:  # First in sequence
        width = struct.unpack('>H', data[7:9])[0]
        height = struct.unpack('>H', data[9:11])[0]
        rle = data[11:]
        return {'width': width, 'height': height, 'rle': rle}
    return None


def decode_image(img_data: dict, palette: dict) -> Image.Image:
    """Decode RLE image data to PIL Image."""
    width = img_data['width']
    height = img_data['height']
    rle = img_data['rle']

    pixels = []
    i = 0
    while i < len(rle) and len(pixels) < width * height:
        byte = rle[i]
        i += 1

        if byte == 0:
            if i >= len(rle):
                break
            flags = rle[i]
            i += 1

            if flags == 0:
                # End of line - pad to width
                while len(pixels) % width != 0:
                    pixels.append(0)
            elif flags & 0xC0 == 0:
                # Short run of color 0
                pixels.extend([0] * flags)
            elif flags & 0xC0 == 0x40:
                # Short run of color 0
                count = flags & 0x3F
                pixels.extend([0] * count)
            elif flags & 0xC0 == 0x80:
                # Short run of non-zero color
                count = flags & 0x3F
                if i >= len(rle):
                    break
                color = rle[i]
                i += 1
                pixels.extend([color] * count)
            elif flags & 0xC0 == 0xC0:
                # Long run
                if i >= len(rle):
                    break
                count = ((flags & 0x3F) << 8) | rle[i]
                i += 1
                if i >= len(rle):
                    break
                color = rle[i]
                i += 1
                pixels.extend([color] * count)
        else:
            pixels.append(byte)

    # Pad if needed
    while len(pixels) < width * height:
        pixels.append(0)

    # Create RGBA image
    img = Image.new('RGBA', (width, height))
    img_pixels = img.load()

    for y in range(height):
        for x in range(width):
            idx = pixels[y * width + x] if y * width + x < len(pixels) else 0
            color = palette.get(idx, (0, 0, 0, 0))
            img_pixels[x, y] = color

    return img


def ocr_image(img: Image.Image) -> str:
    """OCR an image using Tesseract."""
    # Convert to grayscale with white text on black background for better OCR
    gray = Image.new('L', img.size, 0)
    for y in range(img.height):
        for x in range(img.width):
            r, g, b, a = img.getpixel((x, y))
            if a > 128:  # Visible pixel
                # Use luminance
                lum = int(0.299 * r + 0.587 * g + 0.114 * b)
                gray.putpixel((x, y), lum)

    # Invert if needed (white text on black)
    # Scale up for better OCR
    scale = 3
    gray = gray.resize((gray.width * scale, gray.height * scale), Image.Resampling.LANCZOS)

    # OCR
    text = pytesseract.image_to_string(gray, config='--psm 6')
    return text.strip()


def write_srt(subtitles: list[Subtitle], path: Path):
    """Write subtitles to SRT file."""
    with open(path, 'w', encoding='utf-8') as f:
        for i, sub in enumerate(subtitles, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(sub.start_time)} --> {format_timestamp(sub.end_time)}\n")
            f.write(f"{sub.text}\n\n")


def find_pgs_stream(mkv_path: Path) -> int | None:
    """Find PGS subtitle stream index in an MKV file using ffprobe."""
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error',
                '-select_streams', 's',
                '-show_entries', 'stream=index,codec_name',
                '-of', 'csv=p=0',
                str(mkv_path),
            ],
            capture_output=True, text=True, check=True,
        )
    except FileNotFoundError:
        print("Error: ffprobe not found. Install ffmpeg.", file=sys.stderr)
        sys.exit(1)

    for line in result.stdout.strip().splitlines():
        parts = line.strip().split(',')
        if len(parts) >= 2 and parts[1] == 'hdmv_pgs_subtitle':
            return int(parts[0])
    return None


def extract_pgs(mkv_path: Path, sup_path: Path, stream_index: int):
    """Extract PGS subtitle stream from MKV to SUP file."""
    try:
        subprocess.run(
            [
                'ffmpeg', '-y', '-v', 'error',
                '-i', str(mkv_path),
                '-map', f'0:{stream_index}',
                '-c', 'copy',
                str(sup_path),
            ],
            check=True,
        )
    except FileNotFoundError:
        print("Error: ffmpeg not found.", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert PGS/SUP bitmap subtitles to SRT using Tesseract OCR.",
    )
    parser.add_argument('input', help="SUP file (or MKV file with --extract)")
    parser.add_argument('output', nargs='?', help="Output SRT file (default: input with .srt extension)")
    parser.add_argument(
        '--extract', action='store_true',
        help="Input is MKV: extract PGS stream to SUP first, then OCR",
    )
    parser.add_argument(
        '--sup-dir', type=Path, default=None,
        help="Directory for intermediate SUP file (default: same as output)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if args.extract:
        # MKV → SUP → SRT pipeline
        stream_idx = find_pgs_stream(input_path)
        if stream_idx is None:
            print(f"Error: No PGS subtitle stream found in {input_path}", file=sys.stderr)
            sys.exit(1)

        srt_path = Path(args.output) if args.output else input_path.with_suffix('.srt')
        sup_dir = args.sup_dir or srt_path.parent
        sup_path = sup_dir / input_path.with_suffix('.sup').name

        print(f"Extracting PGS stream {stream_idx} from {input_path.name}...")
        extract_pgs(input_path, sup_path, stream_idx)

        print(f"Reading {sup_path}...")
        data = sup_path.read_bytes()
    else:
        # Direct SUP → SRT
        sup_path = input_path
        srt_path = Path(args.output) if args.output else sup_path.with_suffix('.srt')

        print(f"Reading {sup_path}...")
        data = sup_path.read_bytes()

    print("Parsing PGS and running OCR...")
    subtitles = parse_pgs(data)

    print(f"Found {len(subtitles)} subtitles")

    print(f"Writing {srt_path}...")
    write_srt(subtitles, srt_path)

    print("Done!")

    # Preview first few
    print("\nPreview:")
    for sub in subtitles[:5]:
        print(f"  {format_timestamp(sub.start_time)}: {sub.text[:50]}...")


if __name__ == '__main__':
    main()
