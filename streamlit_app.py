"""
Ogrenci Davranis Analiz Sistemi - Streamlit Uygulamasi
=====================================================
Ozellikler:
  1. Canli Kamera ile ogrenci analizi (insan tespiti + siniflandirma)
  2. Resim yukleme ile analiz
  3. PDF -> Sesli Video olusturma
  4. Istatistikler (aktif / pasif oranlari)
"""

import os
import sys
import time
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image as PILImage
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTION_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

HERO_IMAGE_PATH = os.path.join(BASE_DIR, "assets", "hero.jpg")
DINLIYOR_IMAGE_PATH = os.path.join(BASE_DIR, "assets", "dinliyor.jpg")
DINLEMIYOR_IMAGE_PATH = os.path.join(BASE_DIR, "assets", "dinlemiyor.jpg")
PDF_VIDEO_IMAGE_PATH = os.path.join(BASE_DIR, "assets", "pdf_video.jpg")

CLASS_LABELS = {0: "Dinlemiyor", 1: "Dinliyor"}
COLORS = {
    "Dinliyor": (0, 200, 0),
    "Dinlemiyor": (0, 0, 255),
    "Bilinmiyor": (180, 180, 180),
}

# ---------------------------------------------------------------------------
# Sayfa yapilandirmasi
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Ogrenci Davranis Analizi",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Gelismis CSS Stil
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ===== GENEL TEMA ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ===== ANA BASLIK ===== */
    .hero-container {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(74,144,217,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 1; }
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4facfe, #00f2fe, #43e97b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.7);
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 300;
        position: relative;
        z-index: 1;
    }
    .hero-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }

    /* ===== ISTATISTIK KARTLARI ===== */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1.5rem 0;
    }
    @media (max-width: 768px) {
        .stats-grid { grid-template-columns: repeat(2, 1fr); }
    }
    .stat-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    }
    .stat-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #4facfe, #00f2fe);
        border-radius: 16px 16px 0 0;
    }
    .stat-icon { font-size: 1.8rem; margin-bottom: 0.3rem; }
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #fff, #e0e0e0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    .stat-card.aktif::after {
        background: linear-gradient(90deg, #00b09b, #96c93d);
    }
    .stat-card.aktif .stat-number {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-card.pasif::after {
        background: linear-gradient(90deg, #f5515f, #a1051d);
    }
    .stat-card.pasif .stat-number {
        background: linear-gradient(135deg, #f5515f, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-card.oran::after {
        background: linear-gradient(90deg, #f7971e, #ffd200);
    }
    .stat-card.oran .stat-number {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* ===== PROGRESS BAR ===== */
    .progress-container {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        height: 8px;
        margin-top: 1rem;
        overflow: hidden;
    }
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    .progress-green { background: linear-gradient(90deg, #00b09b, #96c93d); }
    .progress-red { background: linear-gradient(90deg, #f5515f, #a1051d); }

    /* ===== MOD BASLIK ===== */
    .section-header {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        padding: 1.2rem 1.5rem;
        border-radius: 14px;
        border-left: 4px solid #4facfe;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .section-header h2 {
        margin: 0;
        font-size: 1.4rem;
        font-weight: 700;
        color: #4facfe;
    }
    .section-header p {
        margin: 0.3rem 0 0 0;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
    }

    /* ===== INFO KUTULARI ===== */
    .info-box {
        background: linear-gradient(145deg, rgba(79,172,254,0.1), rgba(0,242,254,0.05));
        border: 1px solid rgba(79,172,254,0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: #4facfe;
        font-size: 0.9rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(145deg, rgba(255,193,7,0.1), rgba(255,193,7,0.05));
        border: 1px solid rgba(255,193,7,0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: #ffc107;
        font-size: 0.9rem;
        margin: 1rem 0;
    }
    .success-box {
        background: linear-gradient(145deg, rgba(0,176,155,0.1), rgba(150,201,61,0.05));
        border: 1px solid rgba(0,176,155,0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: #00b09b;
        font-size: 0.9rem;
        margin: 1rem 0;
    }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e, #16213e);
    }
    [data-testid="stSidebar"] .stMarkdown h1 {
        background: linear-gradient(90deg, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem;
        text-align: center;
    }

    .sidebar-section {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.8rem 0;
    }
    .sidebar-section-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: rgba(255,255,255,0.4);
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .sidebar-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.3rem 0;
        color: rgba(255,255,255,0.7);
        font-size: 0.85rem;
    }
    .sidebar-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    .badge-green {
        background: rgba(0,176,155,0.2);
        color: #00b09b;
    }
    .badge-red {
        background: rgba(245,81,95,0.2);
        color: #f5515f;
    }

    /* ===== GORSEL CIKTI CERCEVESI ===== */
    .image-frame {
        border: 2px solid rgba(79,172,254,0.2);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    .image-label {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        color: rgba(255,255,255,0.8);
        padding: 0.6rem 1rem;
        font-weight: 600;
        font-size: 0.9rem;
        text-align: center;
        border-bottom: 1px solid rgba(79,172,254,0.15);
    }

    /* ===== FOOTER ===== */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: rgba(255,255,255,0.3);
        font-size: 0.8rem;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 3rem;
    }

    /* ===== BUTON STILI ===== */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4facfe, #00f2fe) !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(79,172,254,0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(79,172,254,0.5) !important;
    }
    .stButton > button:not([kind="primary"]) {
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        transition: all 0.3s ease !important;
    }

    /* ===== DOSYA YUKLEYICI ===== */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(79,172,254,0.3) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
    }

    /* ===== DIVIDER ===== */
    .fancy-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(79,172,254,0.3), transparent);
        margin: 2rem 0;
        border: none;
    }

    /* ===== ROBOT AVATAR ===== */
    .robot-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 1rem;
        background: linear-gradient(145deg, #0f0c29, #1a1a2e);
        border-radius: 20px;
        border: 1px solid rgba(79,172,254,0.15);
        box-shadow: 0 8px 40px rgba(0,0,0,0.3);
        height: 100%;
        min-height: 300px;
    }

    .robot {
        position: relative;
        width: 110px;
        height: 145px;
        animation: robot-float 3s ease-in-out infinite;
        transform: scale(0.8);
    }
    @keyframes robot-float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
    }

    /* Kafa */
    .robot-head {
        width: 85px;
        height: 72px;
        background: linear-gradient(145deg, #3a7bd5, #2b5cad);
        border-radius: 16px;
        margin: 0 auto;
        position: relative;
        box-shadow: 0 6px 18px rgba(58,123,213,0.3), inset 0 2px 4px rgba(255,255,255,0.1);
        border: 2px solid rgba(79,172,254,0.3);
    }

    /* Anten */
    .robot-antenna {
        width: 3px;
        height: 14px;
        background: linear-gradient(to top, #3a7bd5, #4facfe);
        margin: 0 auto;
        border-radius: 2px;
        position: relative;
    }
    .robot-antenna::after {
        content: '';
        position: absolute;
        top: -6px;
        left: -4px;
        width: 10px;
        height: 10px;
        background: radial-gradient(circle, #00f2fe, #4facfe);
        border-radius: 50%;
        animation: antenna-glow 2s ease-in-out infinite;
        box-shadow: 0 0 15px rgba(0,242,254,0.5);
    }
    @keyframes antenna-glow {
        0%, 100% { opacity: 0.5; box-shadow: 0 0 8px rgba(0,242,254,0.3); }
        50% { opacity: 1; box-shadow: 0 0 20px rgba(0,242,254,0.8); }
    }

    /* Gozler */
    .robot-eyes {
        display: flex;
        justify-content: center;
        gap: 16px;
        margin-top: 14px;
    }
    .robot-eye {
        width: 18px;
        height: 18px;
        background: radial-gradient(circle, #fff, #e0f0ff);
        border-radius: 50%;
        position: relative;
        box-shadow: 0 0 12px rgba(255,255,255,0.4);
        animation: eye-blink 4s ease-in-out infinite;
    }
    .robot-eye::after {
        content: '';
        position: absolute;
        width: 8px;
        height: 8px;
        background: radial-gradient(circle, #0a0a2e, #1a1a4e);
        border-radius: 50%;
        top: 5px;
        left: 5px;
        animation: eye-look 5s ease-in-out infinite;
    }
    @keyframes eye-blink {
        0%, 45%, 55%, 100% { transform: scaleY(1); }
        50% { transform: scaleY(0.1); }
    }
    @keyframes eye-look {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(3px); }
        75% { transform: translateX(-3px); }
    }

    /* Agiz — Dogal konusma animasyonu (katmanli) */
    .robot-mouth {
        width: 28px;
        height: 6px;
        background: linear-gradient(135deg, #00f2fe, #4facfe);
        border-radius: 0 0 10px 10px;
        margin: 10px auto 0;
        box-shadow: 0 0 10px rgba(0,242,254,0.3);
        animation:
            mouth-height 0.38s ease-in-out infinite,
            mouth-width 0.53s ease-in-out infinite,
            mouth-radius 0.47s ease-in-out infinite,
            mouth-glow 0.71s ease-in-out infinite;
    }
    /* Yukseklik — ana acilma/kapanma */
    @keyframes mouth-height {
        0%   { height: 4px; }
        12%  { height: 12px; }
        28%  { height: 5px; }
        40%  { height: 15px; }
        55%  { height: 7px; }
        65%  { height: 11px; }
        78%  { height: 4px; }
        88%  { height: 10px; }
        100% { height: 4px; }
    }
    /* Genislik — hafif genisleme */
    @keyframes mouth-width {
        0%   { width: 26px; }
        20%  { width: 34px; }
        35%  { width: 24px; }
        50%  { width: 36px; }
        70%  { width: 28px; }
        85%  { width: 32px; }
        100% { width: 26px; }
    }
    /* Kenar yuvarlaklik — sekil degisimi */
    @keyframes mouth-radius {
        0%   { border-radius: 4px; }
        15%  { border-radius: 0 0 16px 16px; }
        30%  { border-radius: 6px; }
        50%  { border-radius: 0 0 22px 22px; }
        65%  { border-radius: 4px 4px 10px 10px; }
        80%  { border-radius: 0 0 18px 18px; }
        100% { border-radius: 4px; }
    }
    /* Parlaklik — konusma vurgusu */
    @keyframes mouth-glow {
        0%   { box-shadow: 0 0 8px rgba(0,242,254,0.2); }
        25%  { box-shadow: 0 0 20px rgba(0,242,254,0.6); }
        50%  { box-shadow: 0 0 10px rgba(0,242,254,0.25); }
        75%  { box-shadow: 0 0 25px rgba(0,242,254,0.7); }
        100% { box-shadow: 0 0 8px rgba(0,242,254,0.2); }
    }

    /* Govde */
    .robot-body {
        width: 62px;
        height: 48px;
        background: linear-gradient(145deg, #2b5cad, #1e3f7a);
        border-radius: 0 0 16px 16px;
        margin: 3px auto 0;
        position: relative;
        box-shadow: 0 6px 15px rgba(43,92,173,0.3);
        border: 2px solid rgba(79,172,254,0.2);
        border-top: none;
    }
    .robot-body::before {
        content: '';
        position: absolute;
        top: 10px;
        left: 50%;
        transform: translateX(-50%);
        width: 20px;
        height: 20px;
        background: radial-gradient(circle, rgba(0,242,254,0.6), rgba(79,172,254,0.2));
        border-radius: 50%;
        animation: heart-beat 2s ease-in-out infinite;
        box-shadow: 0 0 15px rgba(0,242,254,0.3);
    }
    @keyframes heart-beat {
        0%, 100% { transform: translateX(-50%) scale(1); opacity: 0.6; }
        50% { transform: translateX(-50%) scale(1.15); opacity: 1; }
    }

    /* Kollar */
    .robot-arms {
        display: flex;
        justify-content: space-between;
        width: 100px;
        margin: -36px auto 0;
        pointer-events: none;
    }
    .robot-arm {
        width: 12px;
        height: 36px;
        background: linear-gradient(145deg, #3a7bd5, #2b5cad);
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    .robot-arm.left {
        animation: wave-left 3s ease-in-out infinite;
        transform-origin: top center;
    }
    .robot-arm.right {
        animation: wave-right 2.5s ease-in-out infinite;
        transform-origin: top center;
    }
    @keyframes wave-left {
        0%, 100% { transform: rotate(0deg); }
        50% { transform: rotate(-12deg); }
    }
    @keyframes wave-right {
        0%, 100% { transform: rotate(0deg); }
        25% { transform: rotate(15deg); }
        75% { transform: rotate(5deg); }
    }

    /* Konusma Balonu */
    .speech-bubble {
        background: linear-gradient(145deg, rgba(79,172,254,0.15), rgba(0,242,254,0.08));
        border: 1px solid rgba(79,172,254,0.25);
        border-radius: 12px;
        padding: 0.5rem 0.8rem;
        margin-top: 0.7rem;
        color: rgba(255,255,255,0.85);
        font-size: 0.72rem;
        text-align: center;
        position: relative;
        max-width: 160px;
        animation: bubble-appear 0.5s ease-out;
    }
    .speech-bubble::before {
        content: '';
        position: absolute;
        top: -8px;
        left: 50%;
        transform: translateX(-50%);
        width: 0;
        height: 0;
        border-left: 8px solid transparent;
        border-right: 8px solid transparent;
        border-bottom: 8px solid rgba(79,172,254,0.25);
    }
    @keyframes bubble-appear {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .robot-label {
        color: rgba(255,255,255,0.5);
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero Baslik
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-icon">🎓</div>
    <h1 class="hero-title">Ogrenci Davranis Analiz Sistemi</h1>
    <p class="hero-subtitle">Yapay zeka destekli gercek zamanli ogrenci katilim analizi</p>
</div>
""", unsafe_allow_html=True)


def show_optional_image(path: str, caption: str = None) -> None:
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)


hero_col_left, hero_col_right = st.columns([2, 1], gap="large")
with hero_col_left:
    show_optional_image(HERO_IMAGE_PATH, "Proje genel gorseli")
with hero_col_right:
    st.markdown("""
    <div class="info-box">
        Bu alana ana sayfa gorselini ekleyebilirsin. Dosya yolu:
        <b>assets/hero.jpg</b>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3, gap="large")
with col_a:
    show_optional_image(DINLIYOR_IMAGE_PATH, "Dinliyor ornegi")
    st.caption("Dosya yolu: assets/dinliyor.jpg")
with col_b:
    show_optional_image(DINLEMIYOR_IMAGE_PATH, "Dinlemiyor ornegi")
    st.caption("Dosya yolu: assets/dinlemiyor.jpg")
with col_c:
    show_optional_image(PDF_VIDEO_IMAGE_PATH, "PDF'den video ornegi")
    st.caption("Dosya yolu: assets/pdf_video.jpg")

# ---------------------------------------------------------------------------
# Yardimci fonksiyonlar
# ---------------------------------------------------------------------------

def _find_cls_model() -> str:
    """runs/classify altindaki en guncel best.pt'yi bulur."""
    primary_dir = os.path.join(BASE_DIR, "runs", "classify")
    fallback_dir = os.path.join(BASE_DIR, "runs")
    runs_dir = primary_dir if os.path.isdir(primary_dir) else fallback_dir
    if not os.path.isdir(runs_dir):
        return ""
    candidates = []
    for root, _, files in os.walk(runs_dir):
        if "best.pt" in files:
            path = os.path.join(root, "best.pt")
            candidates.append((os.path.getmtime(path), path))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    return candidates[0][1]


@st.cache_resource
def load_detection_model():
    if not os.path.exists(DETECTION_MODEL_PATH):
        st.error(f"Insan tespit modeli bulunamadi: {DETECTION_MODEL_PATH}")
        st.stop()
    return YOLO(DETECTION_MODEL_PATH)


@st.cache_resource
def load_classification_model():
    path = _find_cls_model()
    if not path:
        return None
    return YOLO(path)


def _to_pil_rgb(image):
    if image is None:
        return None
    if isinstance(image, PILImage.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.size == 0:
            return None
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(rgb).convert("RGB")
    return None


def classify_person_crop(cls_model, crop_bgr: np.ndarray):
    """Kisi kirpmasini siniflandirir."""
    crop_pil = _to_pil_rgb(crop_bgr)
    if crop_pil is None:
        return "Bilinmiyor", 0.0
    try:
        crop_pil = crop_pil.resize((224, 224))
        results = cls_model.predict(source=[crop_pil], verbose=False)
    except Exception as exc:
        st.session_state["cls_error"] = f"Siniflandirma hatasi: {type(exc).__name__}: {exc}"
        return "Bilinmiyor", 0.0

    if results and results[0].probs is not None:
        probs = results[0].probs
        class_id = int(probs.top1)
        confidence = float(probs.top1conf)
        model_names = getattr(cls_model, "names", None)
        if isinstance(model_names, dict) and class_id in model_names:
            label = str(model_names[class_id])
        else:
            label = CLASS_LABELS.get(class_id, f"Bilinmiyor({class_id})")
        return label, confidence

    return "Bilinmiyor", 0.0


def detect_and_classify(
    frame: np.ndarray,
    det_model,
    cls_model,
    conf_threshold: float = 0.5,
    box_thickness: int = 2,
    text_thickness: int = 2,
    font_scale: float = 0.6,
):
    stats = {"Dinliyor": 0, "Dinlemiyor": 0, "Bilinmiyor": 0, "toplam": 0}
    annotated = frame.copy()

    det_results = det_model(frame, verbose=False)
    if not det_results:
        return annotated, stats

    for box in det_results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id != 0 or conf < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 20 or y2 - y1 < 20:
            continue

        crop = frame[y1:y2, x1:x2]
        stats["toplam"] += 1

        if cls_model is not None:
            label, cls_conf = classify_person_crop(cls_model, crop)
            stats[label] = stats.get(label, 0) + 1
            color = COLORS.get(label, (200, 200, 200))
            text = f"{label} ({cls_conf:.0%})"
        else:
            label = "Kisi"
            color = (200, 200, 0)
            text = f"Kisi ({conf:.0%})"

        # Kutu cizimi
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

        # Etiket arka plani — yari saydam
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1 - th - 14), (x1 + tw + 12, y1), color, -1)
        cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

        cv2.putText(
            annotated,
            text,
            (x1 + 6, y1 - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return annotated, stats


def draw_stats_panel(stats: dict) -> None:
    toplam = stats.get("toplam", 0)
    aktif = stats.get("Dinliyor", 0)
    pasif = stats.get("Dinlemiyor", 0)
    oran = (aktif / toplam * 100) if toplam > 0 else 0
    pasif_oran = (pasif / toplam * 100) if toplam > 0 else 0

    st.markdown(f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-icon">👥</div>
            <div class="stat-number">{toplam}</div>
            <div class="stat-label">Toplam Kisi</div>
        </div>
        <div class="stat-card aktif">
            <div class="stat-icon">✅</div>
            <div class="stat-number">{aktif}</div>
            <div class="stat-label">Dinliyor (Aktif)</div>
            <div class="progress-container">
                <div class="progress-bar progress-green" style="width: {oran}%"></div>
            </div>
        </div>
        <div class="stat-card pasif">
            <div class="stat-icon">❌</div>
            <div class="stat-number">{pasif}</div>
            <div class="stat-label">Dinlemiyor (Pasif)</div>
            <div class="progress-container">
                <div class="progress-bar progress-red" style="width: {pasif_oran}%"></div>
            </div>
        </div>
        <div class="stat-card oran">
            <div class="stat-icon">📊</div>
            <div class="stat-number">{oran:.0f}%</div>
            <div class="stat-label">Aktiflik Orani</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# Kenar Cubugu
# ===========================================================================
st.sidebar.markdown("# 🎓 Menu")
st.sidebar.markdown("")

mode = st.sidebar.radio(
    "📋 Islem Secin:",
    [
        "📹 Canli Kamera Analizi",
        "🖼️ Resim Yukle ve Analiz Et",
        "📄 PDF → Sesli Video",
    ],
    label_visibility="collapsed",
)

# Sidebar — Sistem Durumu
st.sidebar.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
cls_path = _find_cls_model()
det_ok = os.path.exists(DETECTION_MODEL_PATH)

st.sidebar.markdown(f"""
<div class="sidebar-section">
    <div class="sidebar-section-title">⚙️ Sistem Durumu</div>
    <div class="sidebar-item">
        🔍 Tespit Modeli
        <span class="sidebar-badge {'badge-green' if det_ok else 'badge-red'}">
            {'Hazir' if det_ok else 'Yok'}
        </span>
    </div>
    <div class="sidebar-item">
        🧠 Siniflandirma Modeli
        <span class="sidebar-badge {'badge-green' if cls_path else 'badge-red'}">
            {'Hazir' if cls_path else 'Yok'}
        </span>
    </div>
    <div class="sidebar-item">
        🏷️ Siniflar: Dinliyor, Dinlemiyor
    </div>
</div>
""", unsafe_allow_html=True)

if cls_path:
    cls_model_info = load_classification_model()
    if cls_model_info is not None:
        names = getattr(cls_model_info, "names", {})
        st.sidebar.markdown(f"""
        <div class="sidebar-section">
            <div class="sidebar-section-title">🧠 Model Detaylari</div>
            <div class="sidebar-item">
                Gorev: {getattr(cls_model_info, 'task', 'bilinmiyor')}
            </div>
            <div class="sidebar-item">
                Etiketler: {names}
            </div>
        </div>
        """, unsafe_allow_html=True)

if "cls_error" in st.session_state:
    st.sidebar.warning(st.session_state["cls_error"])

# Sidebar footer
st.sidebar.markdown("""
<div style="text-align:center; margin-top:2rem; padding:1rem; color:rgba(255,255,255,0.3); font-size:0.75rem;">
    YOLOv8 + Streamlit<br/>
    Ogrenci Davranis Analizi v2.0
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# MOD 1 — CANLI KAMERA ANALIZI
# ===========================================================================
if mode == "📹 Canli Kamera Analizi":
    st.markdown("""
    <div class="section-header">
        <h2>📹 Canli Kamera ile Ogrenci Analizi</h2>
        <p>Kamerayi acin, gercek zamanli olarak ogrencilerin ders dinleme durumunu analiz edin</p>
    </div>
    """, unsafe_allow_html=True)

    if not cls_path:
        st.markdown("""
        <div class="warning-box">
            ⚠️ Siniflandirma modeli bulunamadi. Once <b>finetune.py</b> ile modeli egitin.
        </div>
        """, unsafe_allow_html=True)

    col_settings, col_info = st.columns([1, 2])
    with col_settings:
        st.markdown("##### ⚙️ Ayarlar")
        conf_threshold = st.slider(
            "Guven Esigi", 0.1, 1.0, 0.5, 0.05,
            help="Insan tespiti icin minimum guven degeri"
        )
        camera_idx = st.selectbox(
            "Kamera Secimi", [0, 1, 2], index=0,
            format_func=lambda x: f"Kamera {x}"
        )
        st.markdown("")

    with col_info:
        st.markdown("""
        <div class="info-box">
            ℹ️ <b>Nasil Calisir?</b><br/>
            1. Kameranizi secin ve "Kamerayi Baslat" butonuna basin<br/>
            2. Sistem otomatik olarak kisileri tespit eder<br/>
            3. Her kisi icin "Dinliyor" veya "Dinlemiyor" siniflandirmasi yapar<br/>
            4. Sonuclar gercek zamanli olarak ekranda gosterilir
        </div>
        """, unsafe_allow_html=True)

    col_start, col_stop, _ = st.columns([1, 1, 3])
    with col_start:
        start_btn = st.button("▶️ Kamerayi Baslat", type="primary", use_container_width=True)
    with col_stop:
        stop_btn = st.button("⏹️ Durdur", use_container_width=True)

    if start_btn:
        det_model = load_detection_model()
        cls_model = load_classification_model()

        cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            st.markdown('<div class="warning-box">❌ Kamera acilamadi! Farkli bir kamera secmeyi deneyin.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">✅ Kamera acildi. Analiz basliyor...</div>', unsafe_allow_html=True)

            frame_placeholder = st.empty()
            stats_placeholder = st.empty()
            chart_placeholder = st.empty()

            history = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                annotated, stats = detect_and_classify(
                    frame, det_model, cls_model, conf_threshold
                )

                toplam = stats.get("toplam", 0)
                aktif = stats.get("Dinliyor", 0)
                if toplam > 0:
                    history.append(aktif / toplam * 100)
                else:
                    history.append(0)

                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(
                    frame_rgb, channels="RGB", use_container_width=True
                )

                with stats_placeholder.container():
                    draw_stats_panel(stats)

                if len(history) % 30 == 0 and len(history) > 1:
                    with chart_placeholder.container():
                        st.markdown("##### 📈 Aktiflik Trendi")
                        st.line_chart(history, height=200)

                if stop_btn:
                    break

                time.sleep(0.03)

            cap.release()
            st.markdown('<div class="info-box">📷 Kamera kapatildi.</div>', unsafe_allow_html=True)

            if history:
                st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
                st.markdown("##### 📊 Oturum Ozeti")
                st.line_chart(history, height=250)
                avg = sum(history) / len(history)
                st.metric("🎯 Ortalama Aktiflik", f"{avg:.1f}%")


# ===========================================================================
# MOD 2 — RESIM YUKLEME
# ===========================================================================
elif mode == "🖼️ Resim Yukle ve Analiz Et":
    st.markdown("""
    <div class="section-header">
        <h2>🖼️ Resim ile Ogrenci Analizi</h2>
        <p>Bir sinif fotografi yukleyin, ogrencilerin durumunu aninda analiz edin</p>
    </div>
    """, unsafe_allow_html=True)

    if not cls_path:
        st.markdown("""
        <div class="warning-box">
            ⚠️ Siniflandirma modeli bulunamadi. Once <b>finetune.py</b> ile modeli egitin.
        </div>
        """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "📎 Bir resim secin veya surukleyip birakin",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Desteklenen formatlar: JPG, PNG, BMP, WebP"
    )

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.markdown('<div class="warning-box">❌ Resim okunamadi. Lutfen farkli bir dosya deneyin.</div>', unsafe_allow_html=True)
        else:
            det_model = load_detection_model()
            cls_model = load_classification_model()

            conf_threshold = st.slider(
                "⚙️ Guven Esigi", 0.1, 1.0, 0.5, 0.05, key="img_conf",
                help="Insan tespiti icin minimum guven degeri"
            )

            annotated, stats = detect_and_classify(
                image,
                det_model,
                cls_model,
                conf_threshold,
                box_thickness=4,
                text_thickness=3,
                font_scale=0.8,
            )

            st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class="image-frame">
                    <div class="image-label">📷 Orijinal Goruntu</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )
            with col2:
                st.markdown("""
                <div class="image-frame">
                    <div class="image-label">🔍 Analiz Sonucu</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )

            st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
            draw_stats_panel(stats)

            # Sonuc degerlendirmesi
            toplam = stats.get("toplam", 0)
            aktif = stats.get("Dinliyor", 0)
            if toplam > 0:
                oran = aktif / toplam * 100
                if oran >= 75:
                    msg = "🌟 Harika! Sinifin buyuk cogunlugu dersi aktif olarak dinliyor."
                    box_class = "success-box"
                elif oran >= 50:
                    msg = "👍 Iyi. Yaridan fazlasi aktif, ancak gelistirme alani var."
                    box_class = "info-box"
                else:
                    msg = "⚠️ Dikkat! Sinifin cogunlugu pasif gorunuyor. Etkilesimli aktivite onerilir."
                    box_class = "warning-box"
                st.markdown(f'<div class="{box_class}">{msg}</div>', unsafe_allow_html=True)
            elif toplam == 0:
                st.markdown('<div class="info-box">ℹ️ Resimde kisi tespit edilemedi. Farkli bir resim deneyin veya guven esigini dusurun.</div>', unsafe_allow_html=True)


# ===========================================================================
# MOD 3 — PDF -> SESLI VIDEO
# ===========================================================================
elif mode == "📄 PDF → Sesli Video":
    st.markdown("""
    <div class="section-header">
        <h2>📄 PDF'den Sesli Video Olusturucu</h2>
        <p>PDF dosyanizi yukleyin, Turkce seslendirmeli egitim videosu olusturulsun</p>
    </div>
    """, unsafe_allow_html=True)

    # Ses ayarlari
    col_upload, col_voice = st.columns([2, 1])
    with col_upload:
        uploaded_pdf = st.file_uploader(
            "📎 PDF dosyasi secin veya surukleyip birakin",
            type=["pdf"],
            help="Sadece PDF formati desteklenir"
        )
    with col_voice:
        st.markdown("##### 🎤 Ses Ayarlari")
        voice_options = {
            "👩 Emel (Kadin - Dogal)": "tr-TR-EmelNeural",
            "👨 Ahmet (Erkek - Dogal)": "tr-TR-AhmetNeural",
        }
        selected_voice_label = st.selectbox(
            "Seslendirici",
            list(voice_options.keys()),
            index=0,
            key="voice_select",
            help="Microsoft Neural TTS - dogal insan sesi"
        )
        selected_voice = voice_options[selected_voice_label]

        speed_pct = st.slider(
            "Konusma Hizi", -20, 30, 0, 5,
            key="voice_speed",
            help="Negatif = yavas, Pozitif = hizli",
            format="%+d%%"
        )

    if "pdf_video_path" not in st.session_state:
        st.session_state["pdf_video_path"] = None
    if "pdf_video_name" not in st.session_state:
        st.session_state["pdf_video_name"] = None

    if uploaded_pdf:
        if uploaded_pdf.name != st.session_state["pdf_video_name"]:
            st.session_state["pdf_video_name"] = uploaded_pdf.name
            st.session_state["pdf_video_path"] = None

        file_size_kb = uploaded_pdf.size / 1024
        st.markdown(f"""
        <div class="success-box">
            📄 <b>{uploaded_pdf.name}</b> yuklendi ({file_size_kb:.1f} KB)
        </div>
        """, unsafe_allow_html=True)

        if st.button("🎬 Video Olustur", type="primary"):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
                tmp_pdf.write(uploaded_pdf.getvalue())
                tmp_pdf_path = tmp_pdf.name

            output_path = os.path.join(
                BASE_DIR,
                os.path.splitext(uploaded_pdf.name)[0] + "_video.mp4",
            )

            try:
                sys.path.insert(0, BASE_DIR)
                from pdf_to_video import build_video, DEFAULT_VOICE, DEFAULT_RATE

                # Kullanicinin sectigi sesi ve hizi ayarla
                import pdf_to_video
                pdf_to_video.DEFAULT_VOICE = selected_voice
                pdf_to_video.DEFAULT_RATE = f"{speed_pct:+d}%"

                with st.spinner("🎬 Video olusturuluyor... Bu islem birkac dakika surebilir."):
                    result_path = build_video(tmp_pdf_path, output_path)

                st.session_state["pdf_video_path"] = result_path
                st.markdown('<div class="success-box">✅ Video basariyla olusturuldu!</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Hata: {e}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                if os.path.exists(tmp_pdf_path):
                    os.unlink(tmp_pdf_path)

        if st.session_state["pdf_video_path"] and os.path.exists(
            st.session_state["pdf_video_path"]
        ):
            st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
            st.markdown("##### 🎥 Olusturulan Video")

            col_video, col_robot = st.columns([3, 1])

            with col_video:
                with open(st.session_state["pdf_video_path"], "rb") as vf:
                    st.video(vf.read())

                with open(st.session_state["pdf_video_path"], "rb") as vf:
                    st.download_button(
                        "⬇️ Videoyu Indir",
                        data=vf.read(),
                        file_name=os.path.basename(st.session_state["pdf_video_path"]),
                        mime="video/mp4",
                        use_container_width=True,
                    )

            with col_robot:
                # Secilen sese gore robot mesaji
                voice_name = selected_voice_label.split("(")[0].strip().replace("👩 ", "").replace("👨 ", "")
                st.markdown(f"""
                <div class="robot-container">
                    <div class="robot">
                        <div class="robot-antenna"></div>
                        <div class="robot-head">
                            <div class="robot-eyes">
                                <div class="robot-eye"></div>
                                <div class="robot-eye"></div>
                            </div>
                            <div class="robot-mouth"></div>
                        </div>
                        <div class="robot-body"></div>
                        <div class="robot-arms">
                            <div class="robot-arm left"></div>
                            <div class="robot-arm right"></div>
                        </div>
                    </div>
                    <div class="speech-bubble">
                        🎤 Merhaba! Ben <b>{voice_name}</b>.<br/>Dersi sizin icin seslendiriyorum!
                    </div>
                    <div class="robot-label">AI Seslendirici</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="section-header">
                <h2>📹 Video Izlerken Canli Kamera Analizi</h2>
                <p>Video oynatirken kameranizi acarak ogrenci katilimini olcun</p>
            </div>
            """, unsafe_allow_html=True)

            col_controls, col_preview = st.columns([1, 2])
            with col_controls:
                st.markdown("##### ⚙️ Ayarlar")
                conf_threshold = st.slider(
                    "Guven Esigi", 0.1, 1.0, 0.5, 0.05, key="pdf_conf"
                )
                camera_idx = st.selectbox(
                    "Kamera", [0, 1, 2], index=0, key="pdf_cam",
                    format_func=lambda x: f"Kamera {x}"
                )
                duration_sec = st.slider(
                    "Analiz Suresi (sn)", 5, 120, 20, 5, key="pdf_duration"
                )
                start_cam = st.button(
                    "▶️ Analizi Baslat",
                    type="primary",
                    key="pdf_start",
                    use_container_width=True,
                )

            with col_preview:
                frame_placeholder = st.empty()
                stats_placeholder = st.empty()
                chart_placeholder = st.empty()

            if start_cam:
                det_model = load_detection_model()
                cls_model = load_classification_model()

                cap = cv2.VideoCapture(camera_idx)
                if not cap.isOpened():
                    st.markdown('<div class="warning-box">❌ Kamera acilamadi!</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-box">✅ Kamera acildi. Analiz basliyor...</div>', unsafe_allow_html=True)
                    history = []
                    end_time = time.time() + duration_sec

                    while cap.isOpened() and time.time() < end_time:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        annotated, stats = detect_and_classify(
                            frame, det_model, cls_model, conf_threshold
                        )

                        toplam = stats.get("toplam", 0)
                        aktif = stats.get("Dinliyor", 0)
                        if toplam > 0:
                            history.append(aktif / toplam * 100)
                        else:
                            history.append(0)

                        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(
                            frame_rgb, channels="RGB", use_container_width=True
                        )

                        with stats_placeholder.container():
                            draw_stats_panel(stats)

                        if len(history) % 30 == 0 and len(history) > 1:
                            with chart_placeholder.container():
                                st.markdown("##### 📈 Aktiflik Trendi")
                                st.line_chart(history, height=200)

                        time.sleep(0.03)

                    cap.release()
                    st.markdown('<div class="info-box">📷 Analiz tamamlandi.</div>', unsafe_allow_html=True)

                    if history:
                        st.markdown("##### 📊 Oturum Ozeti")
                        st.line_chart(history, height=250)
                        avg = sum(history) / len(history)
                        st.metric("🎯 Ortalama Aktiflik", f"{avg:.1f}%")


# ===========================================================================
# Footer
# ===========================================================================
st.markdown("""
<div class="footer">
    🎓 Ogrenci Davranis Analiz Sistemi v2.0 — YOLOv8 + Streamlit ile gelistirilmistir
</div>
""", unsafe_allow_html=True)
