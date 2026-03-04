"""
PDF -> Sesli Video Olusturucu
============================
Bir PDF dosyasini alir, slaytlara boler, Turkce seslendirme ekler
ve MP4 video olarak kaydeder.
"""

import os
import re
import textwrap
import tempfile
import asyncio

import pdfplumber
from PIL import Image, ImageDraw, ImageFont
import edge_tts
from moviepy import AudioFileClip, ImageClip, concatenate_videoclips

# Turkce Neural Sesler (Microsoft Edge TTS)
# Kadin sesi: tr-TR-EmelNeural  |  Erkek sesi: tr-TR-AhmetNeural
DEFAULT_VOICE = "tr-TR-EmelNeural"
DEFAULT_RATE = "+0%"     # Hiz: -50% ile +100% arasi  (ornek: "+10%" biraz hizli)
DEFAULT_PITCH = "+0Hz"   # Perde: "-50Hz" ile "+50Hz" arasi

SLIDE_WIDTH = 1280
SLIDE_HEIGHT = 720
BG_COLOR = (30, 30, 60)
TEXT_COLOR = (255, 255, 255)
TITLE_COLOR = (100, 200, 255)
FONT_SIZE = 28
TITLE_FONT_SIZE = 40
MAX_CHARS_PER_SLIDE = 500


def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    font_paths = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    if bold:
        bold_paths = [
            p for p in font_paths if "bold" in p.lower() or "bd" in p.lower() or "b." in p.lower()
        ]
        font_paths = bold_paths + font_paths

    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue

    return ImageFont.load_default()


def extract_pdf_text(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
    return text.strip()


def split_text_to_slides(text: str, max_chars: int = MAX_CHARS_PER_SLIDE):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    slides = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 > max_chars and current:
            slides.append(current.strip())
            current = sentence
        else:
            current += " " + sentence if current else sentence

    if current.strip():
        slides.append(current.strip())

    return slides if slides else ["(Bos icerik)"]


def create_slide_image(text: str, slide_num: int, total_slides: int, save_path: str, title: str = None) -> str:
    img = Image.new("RGB", (SLIDE_WIDTH, SLIDE_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    font = _get_font(FONT_SIZE)
    title_font = _get_font(TITLE_FONT_SIZE, bold=True)

    y = 40

    if title:
        draw.text((SLIDE_WIDTH // 2, y), title, fill=TITLE_COLOR, font=title_font, anchor="mt")
        y += 70

    draw.line([(60, y), (SLIDE_WIDTH - 60, y)], fill=(80, 80, 120), width=2)
    y += 30

    wrapper = textwrap.TextWrapper(width=70)
    lines = []
    for paragraph in text.split("\n"):
        lines.extend(wrapper.wrap(paragraph))
        lines.append("")

    for line in lines:
        if y > SLIDE_HEIGHT - 80:
            break
        draw.text((80, y), line, fill=TEXT_COLOR, font=font)
        y += 36

    footer = f"Slayt {slide_num}/{total_slides}"
    footer_font = _get_font(20)
    draw.text((SLIDE_WIDTH // 2, SLIDE_HEIGHT - 30), footer, fill=(150, 150, 180), font=footer_font, anchor="mt")

    img.save(save_path, "PNG")
    return save_path


def create_special_slide(text: str, save_path: str, color: tuple = TITLE_COLOR) -> str:
    img = Image.new("RGB", (SLIDE_WIDTH, SLIDE_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font = _get_font(48, bold=True)

    lines = text.split("\n")
    total_height = len(lines) * 60
    y = (SLIDE_HEIGHT - total_height) // 2

    for line in lines:
        draw.text((SLIDE_WIDTH // 2, y), line, fill=color, font=font, anchor="mt")
        y += 60

    img.save(save_path, "PNG")
    return save_path


def generate_tts_audio(
    text: str,
    save_path: str,
    voice: str = DEFAULT_VOICE,
    rate: str = DEFAULT_RATE,
    pitch: str = DEFAULT_PITCH,
) -> str:
    """Microsoft Edge Neural TTS ile dogal Turkce ses olusturur."""

    async def _generate():
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            pitch=pitch,
        )
        await communicate.save(save_path)

    # asyncio event loop'u calistir
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(asyncio.run, _generate()).result()
        else:
            loop.run_until_complete(_generate())
    except RuntimeError:
        asyncio.run(_generate())

    return save_path


def build_video(pdf_path: str, output_path: str = None) -> str:
    if output_path is None:
        base = os.path.splitext(pdf_path)[0]
        output_path = base + "_video.mp4"

    print(f"PDF okunuyor: {pdf_path}")
    text = extract_pdf_text(pdf_path)
    if not text:
        raise ValueError("PDF'den metin cikarilamadi")

    slides_text = split_text_to_slides(text)
    total = len(slides_text) + 2
    print(f"{len(slides_text)} slayt olusturulacak")

    clips = []

    with tempfile.TemporaryDirectory() as tmpdir:
        intro_img = create_special_slide(
            "Ders Icerigi\nPDF'den Olusturuldu",
            os.path.join(tmpdir, "intro.png"),
        )
        intro_audio = generate_tts_audio(
            "Ders icerigi basliyor. Iyi seyirler.",
            os.path.join(tmpdir, "intro.mp3"),
        )
        audio_clip = AudioFileClip(intro_audio)
        video_clip = ImageClip(intro_img, duration=audio_clip.duration + 1)
        video_clip = video_clip.with_audio(audio_clip)
        clips.append(video_clip)

        for i, slide_text in enumerate(slides_text, 1):
            print(f"  Slayt {i}/{len(slides_text)} isleniyor...")

            img_path = os.path.join(tmpdir, f"slide_{i}.png")
            audio_path = os.path.join(tmpdir, f"slide_{i}.mp3")

            create_slide_image(slide_text, i + 1, total, img_path, title=f"Bolum {i}")
            generate_tts_audio(slide_text, audio_path)

            a = AudioFileClip(audio_path)
            v = ImageClip(img_path, duration=a.duration + 0.5)
            v = v.with_audio(a)
            clips.append(v)

        outro_img = create_special_slide(
            "Ders Sonu\nTesekkurler!",
            os.path.join(tmpdir, "outro.png"),
        )
        outro_audio = generate_tts_audio(
            "Ders icerigi sona erdi. Tesekkurler.",
            os.path.join(tmpdir, "outro.mp3"),
        )
        a = AudioFileClip(outro_audio)
        v = ImageClip(outro_img, duration=a.duration + 1)
        v = v.with_audio(a)
        clips.append(v)

        print("Video birlestiriliyor...")
        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac", logger="bar")

    print(f"Video kaydedildi: {output_path}")
    return output_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Kullanim: python pdf_to_video.py <pdf_dosyasi>")
        sys.exit(1)
    build_video(sys.argv[1])
