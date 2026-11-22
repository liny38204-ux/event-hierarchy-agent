"""Render a single timestamped image from a chronological short window."""
from io import BytesIO
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile


def write_contact_sheet(frames, destination, *, rows=2, columns=3):
    from PIL import Image, ImageDraw, ImageFont

    if rows < 1 or columns < 1:
        raise ValueError('contact sheet dimensions must be positive')
    if not 1 <= len(frames) <= rows * columns:
        raise ValueError('contact sheet frame count exceeds the requested layout')
    times = [f['time'] for f in frames]
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError('contact sheet frames must be in chronological order')
    thumbnails = []
    for frame in frames:
        with Image.open(frame['path']) as original:
            image = original.convert('RGB')
            image.thumbnail((384, 320), Image.Resampling.LANCZOS)
            thumbnails.append(image)
    # Size the cells for this stream. Labels sit outside the video pixels.
    width = max(220, max(image.width for image in thumbnails))
    height = max(image.height for image in thumbnails)
    header, gap = 30, 3
    sheet = Image.new('RGB', (columns * width + (columns - 1) * gap,
                               rows * (height + header) + (rows - 1) * gap), '#24282d')
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=19)
    except TypeError:  # Pillow 10.0
        font = ImageFont.load_default()
    for index, (frame, image) in enumerate(zip(frames, thumbnails)):
        row, column = divmod(index, columns)
        x, y = column * (width + gap), row * (height + header + gap)
        draw.rectangle((x, y, x + width - 1, y + height + header - 1), fill='#101418')
        draw.text((x + 7, y + 4), f"{frame['id']}   {frame['time']:.3f} s", font=font, fill='white')
        sheet.paste(image, (x + (width - image.width) // 2, y + header + (height - image.height) // 2))
    data = BytesIO()
    sheet.save(data, format='PNG')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
        try:
            tmp.write(data.getbuffer())
            tmp.flush()
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return {'rows': rows, 'columns': columns, 'frame_ids': [f['id'] for f in frames],
            'sha256': hashlib.sha256(data.getbuffer()).hexdigest()}
