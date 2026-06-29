from __future__ import annotations

import io
import os
import re
import tempfile
from datetime import datetime

import easyocr
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from model_utils import (
    analyze_product_fully,
    classify_risk,
    detect_harmful_additives,
    has_sufficient_input,
    load_prediction_models,
    preprocess_batch_excel_data,
)
from ocr_utils import parse_scan_result


st.set_page_config(
    page_title="SMART NutriScan AI",
    page_icon="🧠",
    layout="wide",
)

# --- BILINGUAL HELPER ---
if "lang" not in st.session_state:
    st.session_state.lang = "ID"

def t(id_text, en_text):
    """Fungsi helper untuk terjemahan string UI."""
    return id_text if st.session_state.lang == "ID" else en_text

def tr_risk(label):
    """Menerjemahkan label risiko khusus bahasa Inggris."""
    if st.session_state.lang == "ID":
        return label
    mapping = {"Aman": "Safe", "Sedang": "Moderate", "Tinggi": "High", "Belum dianalisis": "Not analyzed"}
    return mapping.get(label, label)


if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

if "batch_result_df" not in st.session_state:
    st.session_state.batch_result_df = None
    st.session_state.batch_total_rows = 0


@st.cache_resource(show_spinner=False)
def load_all_models_and_scaler():
    return load_prediction_models()


@st.cache_resource(show_spinner=False)
def load_ocr_model():
    return easyocr.Reader(["id", "en"], gpu=False)


feat_model, lgbm_model, w2v_model, scaler = load_all_models_and_scaler()

def get_ocr_reader_safely():
    try:
        return load_ocr_model(), None
    except Exception as exc:
        return None, str(exc)


def standardize_image_size(image, target_ratio=4/3):
    width, height = image.size
    current_ratio = width / height

    if current_ratio > target_ratio:
        new_width = width
        new_height = int(width / target_ratio)
    else:
        new_height = height
        new_width = int(height * target_ratio)

    new_img = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
    image = image.convert("RGBA")
    new_img.paste(image, ((new_width - width) // 2, (new_height - height) // 2))
    return new_img


def safe_image(image, caption=None, width=None):
    try:
        if width:
            st.image(image, caption=caption, width=width)
        else:
            st.image(image, caption=caption, use_container_width=True)
    except TypeError:
        if width:
            st.image(image, caption=caption, width=width)
        else:
            st.image(image, caption=caption, use_column_width=True)


def fmt(value, digits=2, suffix=""):
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:
        return f"0.{'0' * digits}{suffix}"


def run_ocr_safely(reader, image, mode):
    try:
        return parse_scan_result(reader, image, mode=mode), None
    except Exception as exc:
        return None, str(exc)


OCR_WIDGET_KEY_MAP = {
    "takaran_saji": "ocr_saji",
    "energi": "ocr_energi",
    "lemak_total": "ocr_lemak",
    "lemak_jenuh": "ocr_jenuh",
    "protein": "ocr_protein",
    "karbohidrat": "ocr_karbo",
    "gula": "ocr_gula",
    "garam": "ocr_garam",
    "natrium": "ocr_natrium",
    "natrium_benzoat": "ocr_benzoat",
    "komposisi": "ocr_komposisi",
    "product_name": "ocr_name",
}


LEGACY_OCR_WIDGET_KEYS = set(OCR_WIDGET_KEY_MAP.values())

for _legacy_key in list(LEGACY_OCR_WIDGET_KEYS):
    if _legacy_key in st.session_state:
        del st.session_state[_legacy_key]


def set_widget_default(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def sync_ocr_value_to_form(key, value):
    if key not in st.session_state.ocr_data:
        return
    st.session_state.ocr_data[key] = value


def bump_ocr_form_version():
    st.session_state.ocr_form_version += 1


def clear_ocr_analysis_result():
    st.session_state.ocr_analysis_result = None


def render_ocr_result_debug(scan_result, label):
    if not scan_result:
        return

    errors = scan_result.get("errors", [])
    quality_warnings = scan_result.get("quality_warnings", [])

    if quality_warnings:
        with st.expander(t(f"Catatan kualitas OCR {label}", f"OCR quality notes for {label}"), expanded=False):
            for item in quality_warnings:
                st.warning(item)

    if errors:
        with st.expander(t(f"Catatan error OCR {label}", f"OCR error notes for {label}"), expanded=False):
            for item in errors:
                st.warning(item)

    with st.expander(t(f"Lihat teks OCR {label}", f"View OCR text for {label}"), expanded=False):
        st.caption(t(f"Variasi gambar terbaik: {scan_result.get('best_variant', 'tidak diketahui')}", f"Best image variant: {scan_result.get('best_variant', 'unknown')}"))
        st.text(scan_result.get("raw_text") or t("Tidak ada teks terbaca", "No text readable"))

    with st.expander(t(f"Ringkasan variasi preprocessing {label}", f"Preprocessing variants summary for {label}"), expanded=False):
        names = list(scan_result.get("variants", {}).keys())
        st.write(names if names else t("Tidak ada variasi gambar tersimpan.", "No image variants saved."))


NUTRITION_KEYS = [
    "energi",
    "lemak_total",
    "lemak_jenuh",
    "protein",
    "karbohidrat",
    "gula",
    "garam",
    "natrium",
    "natrium_benzoat",
]

EXAMPLE_PRESETS = {
    "Kosong": {
        "product_name": "",
        "takaran_saji": 100.0,
        "energi": 0.0,
        "lemak_total": 0.0,
        "lemak_jenuh": 0.0,
        "protein": 0.0,
        "karbohidrat": 0.0,
        "gula": 0.0,
        "garam": 0.0,
        "natrium": 0.0,
        "natrium_benzoat": 0.0,
        "komposisi": "",
    },
    "Contoh Aman (Susu Gandum)": {
        "product_name": "Susu Gandum Murni",
        "takaran_saji": 200.0,
        "energi": 120.0,
        "lemak_total": 3.0,
        "lemak_jenuh": 1.0,
        "protein": 6.0,
        "karbohidrat": 15.0,
        "gula": 3.0,
        "garam": 0.25,
        "natrium": 100.0,
        "natrium_benzoat": 0.0,
        "komposisi": "Air, gandum utuh, susu segar, ekstrak malt, sedikit gula tebu, garam laut.",
    },
    "Contoh Sedang (Biskuit Cokelat)": {
        "product_name": "Biskuit Cokelat Renyah",
        "takaran_saji": 50.0,
        "energi": 250.0,
        "lemak_total": 12.0,
        "lemak_jenuh": 5.0,
        "protein": 4.0,
        "karbohidrat": 30.0,
        "gula": 18.0,
        "garam": 1.0,
        "natrium": 400.0,
        "natrium_benzoat": 50.0,
        "komposisi": "Tepung terigu, gula, lemak reroti, cokelat bubuk, susu bubuk, pengembang, perisa sintetik cokelat, pengawet kalium sorbat.",
    },
    "Contoh Tinggi (Keripik Ekstra Pedas)": {
        "product_name": "Keripik Ekstra Pedas",
        "takaran_saji": 100.0,
        "energi": 500.0,
        "lemak_total": 25.0,
        "lemak_jenuh": 12.0,
        "protein": 5.0,
        "karbohidrat": 60.0,
        "gula": 40.0,
        "garam": 3.0,
        "natrium": 1200.0,
        "natrium_benzoat": 200.0,
        "komposisi": "Jagung, minyak nabati terhidrogenasi, gula, sirup fruktosa, bumbu pedas (mengandung mononatrium glutamat, pewarna sintetik kuning FCF, pemanis buatan aspartam), pengawet natrium benzoat.",
    },
}

EN_PRESET_NAMES = {
    "Kosong": "Empty",
    "Contoh Aman (Susu Gandum)": "Safe Example (Oat Milk)",
    "Contoh Sedang (Biskuit Cokelat)": "Moderate Example (Chocolate Biscuit)",
    "Contoh Tinggi (Keripik Ekstra Pedas)": "High Example (Extra Spicy Chips)"
}

def apply_manual_preset():
    preset_name = st.session_state.preset_selector
    defaults = EXAMPLE_PRESETS[preset_name]
    
    st.session_state["manual_name"] = defaults["product_name"]
    st.session_state["manual_saji"] = defaults["takaran_saji"]
    st.session_state["manual_energi"] = defaults["energi"]
    st.session_state["manual_lemak"] = defaults["lemak_total"]
    st.session_state["manual_jenuh"] = defaults["lemak_jenuh"]
    st.session_state["manual_protein"] = defaults["protein"]
    st.session_state["manual_karbo"] = defaults["karbohidrat"]
    st.session_state["manual_gula"] = defaults["gula"]
    st.session_state["manual_garam"] = defaults["garam"]
    st.session_state["manual_natrium"] = defaults["natrium"]
    st.session_state["manual_benzoat"] = defaults["natrium_benzoat"]
    st.session_state["manual_komposisi"] = defaults["komposisi"]


def init_parsed_data():
    return dict(EXAMPLE_PRESETS["Kosong"])


if "ocr_data" not in st.session_state:
    st.session_state.ocr_data = init_parsed_data()

if "ocr_form_version" not in st.session_state:
    st.session_state.ocr_form_version = 0

if "ocr_analysis_result" not in st.session_state:
    st.session_state.ocr_analysis_result = None

if "manual_analysis_result" not in st.session_state:
    st.session_state.manual_analysis_result = None


def hitung_tdee_dinamis(gender, usia, berat, tinggi, aktivitas):
    if gender in ["Pria", "Male"]:
        bmr = (10 * berat) + (6.25 * tinggi) - (5 * usia) + 5
    else:
        bmr = (10 * berat) + (6.25 * tinggi) - (5 * usia) - 161

    faktor = {
        "Sedentary": 1.2,
        "Ringan": 1.375, "Light": 1.375,
        "Sedang": 1.55, "Moderate": 1.55,
        "Aktif": 1.725, "Active": 1.725,
        "Sangat Aktif": 1.9, "Very Active": 1.9,
    }

    tdee = bmr * faktor.get(aktivitas, 1.2)
    return {
        "kalori": tdee,
        "gula": (tdee * 0.10) / 4,
        "lemak_jenuh": (tdee * 0.10) / 9,
        "natrium": 2000,
    }


def build_nutrition_data(
    energi, lemak_total, lemak_jenuh, protein, karbohidrat, gula, garam, natrium, natrium_benzoat
):
    return {
        "energi": float(energi),
        "lemak_total": float(lemak_total),
        "lemak_jenuh": float(lemak_jenuh),
        "protein": float(protein),
        "karbohidrat": float(karbohidrat),
        "gula": float(gula),
        "garam": float(garam),
        "natrium": float(natrium),
        "natrium_benzoat": float(natrium_benzoat),
    }


def render_risk_status(risk_score):
    risk_info = classify_risk(risk_score)
    label_tr = tr_risk(risk_info["label"])
    
    st.metric(t("Skor Risiko Prediksi", "Predicted Risk Score"), f"{risk_score:.2f}%")
    st.metric(t("Klasifikasi", "Classification"), label_tr)

    if risk_info["style"] == "success":
        st.success(f"{t('Klasifikasi', 'Classification')} {label_tr}")
    elif risk_info["style"] == "warning":
        st.warning(f"{t('Klasifikasi', 'Classification')} {label_tr}")
    else:
        st.error(f"{t('Klasifikasi', 'Classification')} {label_tr}")


def render_xai_radar(xai_factors):
    categories = list(xai_factors.keys())
    if not categories:
        return

    norm_values = []
    for key, value in xai_factors.items():
        key_lower = key.lower()
        value = float(value or 0)
        if "gula" in key_lower or "sugar" in key_lower:
            norm_values.append(min((value / 45) * 100, 100))
        elif "natrium" in key_lower and "benzoat" not in key_lower:
            norm_values.append(min((value / 1500) * 100, 100))
        elif "benzoat" in key_lower:
            norm_values.append(min((value / 300) * 100, 100))
        elif "lemak total" in key_lower or "fat" in key_lower:
            norm_values.append(min((value / 30) * 100, 100))
        elif "lemak jenuh" in key_lower or "saturated" in key_lower:
            norm_values.append(min((value / 14) * 100, 100))
        elif "energi" in key_lower or "energy" in key_lower:
            norm_values.append(min((value / 550) * 100, 100))
        elif "karbohidrat" in key_lower or "carbo" in key_lower:
            norm_values.append(min((value / 75) * 100, 100))
        else:
            norm_values.append(min((value / 100) * 100, 100))

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=norm_values + [norm_values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(79, 70, 229, 0.4)",
            line=dict(color="#4F46E5", width=2.5),
            marker=dict(symbol="circle", size=8, color="#312E81"),
            name=t("Kandungan Produk", "Product Content"),
            hoverinfo="r+theta"
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, 100], 
                showticklabels=False,
                gridcolor="rgba(200, 200, 200, 0.3)",
                linecolor="rgba(200, 200, 200, 0.3)"
            ),
            angularaxis=dict(
                gridcolor="rgba(200, 200, 200, 0.3)",
                linecolor="rgba(200, 200, 200, 0.3)",
            ),
            bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=False,
        height=350,
        margin=dict(l=40, r=40, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_recommendation_details(risk_info, recommendation_text, is_upf, upf_flags):
    st.markdown(t("### Rekomendasi", "### Recommendation"))
    
    if risk_info["style"] == "success":
        st.info(f"{recommendation_text}")
    elif risk_info["style"] == "warning":
        st.warning(f"{recommendation_text}")
    else:
        st.error(f"{recommendation_text}")

    if is_upf:
        st.error(t("Indikasi bahan ultra proses terdeteksi", "Ultra-processed food ingredients detected"))
        st.write(", ".join(upf_flags))
        
    st.markdown("---")
    
    with st.expander(t("ℹ️ Detail Penjelasan Klasifikasi Nutrisi", "ℹ️ Nutrition Classification Details"), expanded=False):
        st.markdown(t("""
        * 🟢 **Aman (0 - 34.99):** Produk relatif aman dan sehat. Cocok untuk dikonsumsi dalam porsi wajar sebagai bagian dari asupan nutrisi harian Anda.
        * 🟡 **Sedang (35 - 69.99):** Kandungan produk memiliki beberapa catatan (misal: kalori cukup padat atau ada gula tambahan). Boleh dikonsumsi sesekali, namun bukan untuk konsumsi utama harian yang berulang-ulang.
        * 🔴 **Tinggi (70 - 100):** Sangat disarankan untuk dibatasi. Produk ini kemungkinan besar padat energi tanpa nutrisi bermanfaat (empty calories), tinggi gula/garam, atau merupakan produk *ultra-processed*.
        """, """
        * 🟢 **Safe (0 - 34.99):** Product is relatively safe and healthy. Suitable for consumption in reasonable portions as part of your daily nutritional intake.
        * 🟡 **Moderate (35 - 69.99):** Product content has some remarks (e.g., quite calorie-dense or added sugars). Okay to consume occasionally, but not as a repeated daily staple.
        * 🔴 **High (70 - 100):** Highly recommended to limit. This product is likely energy-dense without beneficial nutrients (empty calories), high in sugar/salt, or is an *ultra-processed* product.
        """))


def render_nutrition_kepadatan_gula(nutrition_data, takaran_saji):
    energi = float(nutrition_data.get("energi", 0))
    gula = float(nutrition_data.get("gula", 0))
    karbohidrat = float(nutrition_data.get("karbohidrat", 0))

    kepadatan = energi / takaran_saji if takaran_saji > 0 else 0
    rasio_gula = (gula / karbohidrat * 100) if karbohidrat > 0 else 0

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(t("**Kepadatan Energi (kkal/gram)**", "**Energy Density (kcal/gram)**"))
        st.markdown(f"## {kepadatan:.1f}")
        if kepadatan > 4.0:
            st.error(t("↑ 🔴 Sangat Tinggi (Padat Kalori)", "↑ 🔴 Very High (Calorie Dense)"))
            st.caption(t("Menunjukkan seberapa padat kalori dalam produk ini. Kepadatan tinggi memicu obesitas jika tidak dikontrol.", "Indicates how calorie-dense the product is. High density triggers obesity if uncontrolled."))
        elif kepadatan >= 1.5:
            st.warning(t("— 🟡 Sedang", "— 🟡 Moderate"))
            st.caption(t("Kepadatan kalori moderat. Perhatikan porsi konsumsi Anda.", "Moderate calorie density. Watch your consumption portion."))
        else:
            st.success(t("↓ 🟢 Rendah Kalori", "↓ 🟢 Low Calorie"))
            st.caption(t("Produk ini memiliki kepadatan energi yang rendah, baik untuk mengontrol asupan kalori.", "This product has low energy density, good for controlling calorie intake."))

    with col2:
        st.markdown(t("**Rasio Gula dari Total Karbohidrat**", "**Sugar Ratio of Total Carbohydrates**"))
        st.markdown(f"## {rasio_gula:.1f}%")
        if rasio_gula > 50:
            st.error(t("↑ 🔴 Tinggi Gula Sederhana", "↑ 🔴 High in Simple Sugars"))
            st.caption(t("Jika >50%, sebagian besar karbohidrat adalah gula sederhana yang bisa memicu lonjakan gula darah (*sugar spike*).", "If >50%, most carbohydrates are simple sugars that can trigger a blood sugar spike."))
        elif rasio_gula >= 20:
            st.warning(t("— 🟡 Sedang", "— 🟡 Moderate"))
            st.caption(t("Mengandung gula sederhana dalam jumlah sedang.", "Contains simple sugars in moderate amounts."))
        else:
            st.success(t("↓ 🟢 Rendah Gula", "↓ 🟢 Low Sugar"))
            st.caption(t("Sebagian besar karbohidrat berasal dari sumber kompleks yang lebih lama dicerna.", "Most carbohydrates come from complex sources that digest slower."))


def render_nutrition_pie_chart(nutrition_data):
    lemak_total = float(nutrition_data.get("lemak_total", 0))
    karbohidrat = float(nutrition_data.get("karbohidrat", 0))
    protein = float(nutrition_data.get("protein", 0))

    kalori_lemak = lemak_total * 9
    kalori_karbo = karbohidrat * 4
    kalori_protein = protein * 4
    total_kal_makro = kalori_lemak + kalori_karbo + kalori_protein

    if total_kal_makro > 0:
        labels = [
            t('Lemak (9 kkal/g)', 'Fat (9 kcal/g)'), 
            t('Karbohidrat (4 kkal/g)', 'Carbohydrate (4 kcal/g)'), 
            t('Protein (4 kkal/g)', 'Protein (4 kcal/g)')
        ]
        values = [kalori_lemak, kalori_karbo, kalori_protein]
        colors = ['#E74C3C', '#2ECC71', '#3498DB'] 

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, marker=dict(colors=colors))])
        fig.update_traces(textinfo='percent+label', textposition='inside')
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(t("Data makronutrien kosong atau bernilai nol. Isi Lemak, Karbohidrat, dan Protein untuk melihat rasio kalori.", "Macronutrient data is empty or zero. Fill in Fat, Carbohydrate, and Protein to see calorie ratios."))


def render_holistic_nutrition_profile(nutrition_data, takaran_saji):
    st.markdown(t("### 📊 Profil Gizi & Makronutrien Holistik", "### 📊 Holistic Nutrition & Macronutrient Profile"))
    st.caption(t("Analisis mendalam mengenai sumber kalori dan dampak glikemik berdasarkan takaran saji.", "In-depth analysis of calorie sources and glycemic impact based on serving size."))
    render_nutrition_kepadatan_gula(nutrition_data, takaran_saji)
    st.write("")
    st.markdown(t("**Distribusi Sumber Kalori (Macronutrient Split)**", "**Calorie Source Distribution (Macronutrient Split)**"))
    render_nutrition_pie_chart(nutrition_data)


def custom_progress_bar(label, current_val, max_val, unit, color, percentage):
    display_pct = min(percentage, 100)
    
    warning_text = ""
    if percentage > 100:
        color = "#E74C3C" 
        w_txt = t("Melebihi Batas!", "Exceeds Limit!")
        warning_text = f"<span style='color:#E74C3C; font-weight:bold; font-size: 0.9em; margin-left: 5px;'>({w_txt})</span>"

    html_code = f"<div style='margin-bottom: 24px; font-family: \"Inter\", \"Segoe UI\", sans-serif;'><div style='display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 8px;'><span style='font-weight: 600; font-size: 15px; color: #0F172A;'>{label}</span><span style='color: #475569; font-size: 14px;'><span style='font-weight: 700; color: #1E293B;'>{current_val:.2f}</span> / {max_val:.2f} {unit} <span style='color: #64748B; margin-left: 4px;'>({percentage:.1f}%)</span>{warning_text}</span></div><div style='width: 100%; background-color: #E2E8F0; border-radius: 8px; height: 14px; box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);'><div style='width: {display_pct}%; background-color: {color}; height: 100%; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.1); transition: width 0.8s ease-out;'></div></div></div>"
    st.markdown(html_code, unsafe_allow_html=True)


def render_health_metrics(nutrition_data, takaran_saji, current_threshold, show_header=True):
    if show_header:
        st.markdown(t("### 🎯 Pemenuhan Angka Kecukupan Gizi Harian", "### 🎯 Fulfillment of Daily Nutritional Adequacy"))
        st.caption(t("Berdasarkan profil pengguna dan batas ambang kesehatan medis Anda:", "Based on your user profile and medical health thresholds:"))
        st.write("")

    gula = float(nutrition_data.get("gula", 0))
    natrium = float(nutrition_data.get("natrium", 0))
    lemak_jenuh = float(nutrition_data.get("lemak_jenuh", 0))

    gula_pct = (gula / current_threshold["gula"] * 100) if current_threshold["gula"] else 0
    natrium_pct = (natrium / current_threshold["natrium"] * 100) if current_threshold["natrium"] else 0
    lemak_jenuh_pct = (lemak_jenuh / current_threshold["lemak_jenuh"] * 100) if current_threshold["lemak_jenuh"] else 0

    custom_progress_bar(t("Gula", "Sugar"), gula, current_threshold["gula"], "g", "#F59E0B", gula_pct)
    custom_progress_bar(t("Natrium", "Sodium"), natrium, current_threshold["natrium"], "mg", "#3498DB", natrium_pct)
    custom_progress_bar(t("Lemak Jenuh", "Saturated Fat"), lemak_jenuh, current_threshold["lemak_jenuh"], "g", "#9B59B6", lemak_jenuh_pct)


def make_analysis_signature(product_name, takaran_saji, nutrition_data, komposisi):
    return {
        "product_name": str(product_name or "").strip(),
        "takaran_saji": round(float(takaran_saji or 0), 4),
        "nutrition": {
            key: round(float(nutrition_data.get(key, 0) or 0), 4)
            for key in sorted(nutrition_data.keys())
        },
        "komposisi": str(komposisi or "").strip(),
    }


def build_analysis_result(product_name, takaran_saji, nutrition_data, komposisi):
    if not has_sufficient_input(nutrition_data):
        return {
            "status": "insufficient",
            "message": t("Data belum cukup untuk dianalisis. Isi atau koreksi minimal satu nilai gizi yang valid sebelum menjalankan rekomendasi.", "Insufficient data for analysis. Fill in or correct at least one valid nutritional value before running recommendations."),
            "integrity_note": t("Sistem tidak memberi label tinggi, sedang, atau aman ketika data masih kosong. Ini menjaga integritas hasil analisis.", "System does not label high, moderate, or safe when data is empty. This preserves analysis integrity."),
            "product_name": product_name or t("Produk Tanpa Nama", "Unnamed Product"),
            "takaran_saji": float(takaran_saji or 0),
            "nutrition_data": nutrition_data,
            "komposisi": komposisi,
        }

    risk_score, xai_factors, recommendation = analyze_product_fully(
        nutrition_data,
        komposisi,
        feat_model,
        lgbm_model,
        w2v_model,
        scaler,
    )
    risk_info = classify_risk(risk_score)
    is_upf, flags = detect_harmful_additives(komposisi)

    return {
        "status": "ok",
        "product_name": product_name or t("Produk Tanpa Nama", "Unnamed Product"),
        "takaran_saji": float(takaran_saji or 0),
        "nutrition_data": nutrition_data,
        "komposisi": komposisi,
        "risk_score": float(risk_score),
        "risk_info": risk_info,
        "xai_factors": xai_factors,
        "recommendation": recommendation,
        "is_upf": is_upf,
        "upf_flags": flags,
    }


def generate_batch_insights(df_results):
    if df_results is None or df_results.empty:
        return t("Belum ada data untuk dianalisis.", "No data to analyze yet.")

    valid_df = df_results.copy()
    valid_df["Skor Risiko Numerik"] = pd.to_numeric(valid_df["Skor Risiko"], errors='coerce')
    valid_df = valid_df.dropna(subset=["Skor Risiko Numerik"])

    if valid_df.empty:
        return t("Data tidak valid untuk membuat ringkasan AI.", "Data is invalid for creating AI summary.")

    total_products = len(valid_df)
    aman_count = len(valid_df[valid_df["Klasifikasi"] == "Aman"])
    pct_aman = (aman_count / total_products) * 100 if total_products > 0 else 0

    highest_risk_row = valid_df.loc[valid_df["Skor Risiko Numerik"].idxmax()]
    highest_name = highest_risk_row.get("Nama Produk", t("Produk Tidak Diketahui", "Unknown Product"))
    highest_score = highest_risk_row["Skor Risiko Numerik"]

    avg_score = valid_df["Skor Risiko Numerik"].mean()
    if avg_score < 35:
        avg_cat = t("rendah hingga sedang", "low to moderate")
    elif avg_score < 70:
        avg_cat = t("sedang hingga tinggi", "moderate to high")
    else:
        avg_cat = t("tinggi", "high")

    reason_text = t("kandungan komposisi gizinya", "its nutritional composition")
    if "nutrition_data" in highest_risk_row and isinstance(highest_risk_row["nutrition_data"], dict):
        nut_data = highest_risk_row["nutrition_data"]
        high_factors = []
        
        if float(nut_data.get("natrium", 0)) > 300: high_factors.append("natrium")
        if float(nut_data.get("gula", 0)) > 12: high_factors.append("gula")
        if float(nut_data.get("lemak_total", 0)) > 10: high_factors.append(t("lemak", "fat"))
        if float(nut_data.get("lemak_jenuh", 0)) > 4: high_factors.append(t("lemak jenuh", "saturated fat"))
        
        if high_factors:
            if len(high_factors) > 1:
                dan = t("dan", "and")
                reason_text = t(f"kandungan {', '.join(high_factors[:-1])} {dan} {high_factors[-1]} yang relatif tinggi", 
                                f"relatively high {', '.join(high_factors[:-1])} {dan} {high_factors[-1]} content")
            else:
                reason_text = t(f"kandungan {high_factors[0]} yang relatif tinggi", 
                                f"relatively high {high_factors[0]} content")

    insight = t(
        f"Dari **{total_products} produk** yang dianalisis, **{pct_aman:.1f}%** termasuk kategori aman. "
        f"Produk dengan skor risiko tertinggi adalah **{highest_name}** ({highest_score:.2f}), "
        f"terutama dipengaruhi oleh {reason_text}. "
        f"Secara keseluruhan, rata-rata skor risiko batch ini adalah **{avg_score:.1f}** yang menunjukkan "
        f"tingkat risiko konsumsi berada pada kategori **{avg_cat}**.",
        
        f"Out of **{total_products} products** analyzed, **{pct_aman:.1f}%** fall into the safe category. "
        f"The product with the highest risk score is **{highest_name}** ({highest_score:.2f}), "
        f"primarily driven by {reason_text}. "
        f"Overall, the average risk score for this batch is **{avg_score:.1f}**, indicating "
        f"the consumption risk level is in the **{avg_cat}** category."
    )
    return insight


def generate_pdf_report(df_results, insight_text, fig_pie):
    from fpdf import FPDF
    
    class PDFReport(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 16)
            self.set_text_color(15, 23, 42)
            self.cell(0, 10, t('Laporan Analisis Batch - SMART NutriScan AI', 'Batch Analysis Report - SMART NutriScan AI'), 0, 1, 'C')
            self.set_draw_color(200, 200, 200)
            self.line(10, 22, 200, 22)
            self.ln(10)
            
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, t(f'Halaman {self.page_no()}', f'Page {self.page_no()}'), 0, 0, 'C')

    pdf = PDFReport()
    pdf.add_page()
    
    # 1. Ringkasan Eksekutif
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 10, t("1. Ringkasan Eksekutif", "1. Executive Summary"), 0, 1)
    
    pdf.set_font("Arial", '', 11)
    pdf.set_text_color(50, 50, 50)
    clean_insight = insight_text.replace('**', '').replace('\n', ' ')
    clean_insight = clean_insight.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 7, clean_insight)
    pdf.ln(5)
    
    # 2. Statistik
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 10, t("2. Statistik Distribusi Risiko", "2. Risk Distribution Statistics"), 0, 1)
    
    valid_df = df_results.dropna(subset=["Skor Risiko Numerik"])
    total = len(valid_df)
    
    if total > 0:
        aman = len(valid_df[valid_df["Klasifikasi"] == "Aman"])
        sedang = len(valid_df[valid_df["Klasifikasi"] == "Sedang"])
        tinggi = len(valid_df[valid_df["Klasifikasi"] == "Tinggi"])
        avg_score = valid_df["Skor Risiko Numerik"].mean()
        
        pdf.set_font("Arial", '', 11)
        pdf.set_text_color(50, 50, 50)
        stats_text = t(
            f"- Total Produk Dianalisis: {total} produk\n"
            f"- Kategori Aman: {aman} produk ({(aman/total)*100:.1f}%)\n"
            f"- Kategori Sedang: {sedang} produk ({(sedang/total)*100:.1f}%)\n"
            f"- Kategori Tinggi: {tinggi} produk ({(tinggi/total)*100:.1f}%)\n"
            f"- Rata-rata Skor Risiko Keseluruhan: {avg_score:.2f} / 100",
            
            f"- Total Analyzed Products: {total} products\n"
            f"- Safe Category: {aman} products ({(aman/total)*100:.1f}%)\n"
            f"- Moderate Category: {sedang} products ({(sedang/total)*100:.1f}%)\n"
            f"- High Category: {tinggi} products ({(tinggi/total)*100:.1f}%)\n"
            f"- Overall Average Risk Score: {avg_score:.2f} / 100"
        )
        pdf.multi_cell(0, 7, stats_text)
    pdf.ln(5)
    
    # 3. Grafik Proporsi Klasifikasi
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 10, t("3. Grafik Proporsi Klasifikasi", "3. Classification Proportion Chart"), 0, 1)
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            fig_pie.write_image(tmpfile.name, format="png", width=600, height=400)
            pdf.image(tmpfile.name, x=45, w=120)
        os.unlink(tmpfile.name)
        pdf.ln(5)
    except Exception as e:
        pdf.set_font("Arial", 'I', 10)
        pdf.set_text_color(200, 50, 50)
        pdf.cell(0, 10, t("(Grafik tidak dapat diekspor. Pastikan package 'kaleido' terinstall di backend.)", "(Chart cannot be exported. Ensure 'kaleido' package is installed in backend.)"), 0, 1)
        pdf.ln(5)
        
    # 4. Tabel Hasil Analisis
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 10, t("4. Detail Tabel Hasil Analisis", "4. Detailed Analysis Results Table"), 0, 1)
    pdf.ln(2)
    
    # Table Header
    pdf.set_font("Arial", 'B', 10)
    pdf.set_fill_color(240, 244, 248)
    col_w = [110, 35, 45]
    pdf.cell(col_w[0], 10, t("Nama Produk", "Product Name"), 1, 0, 'C', fill=True)
    pdf.cell(col_w[1], 10, t("Skor Risiko", "Risk Score"), 1, 0, 'C', fill=True)
    pdf.cell(col_w[2], 10, t("Klasifikasi", "Classification"), 1, 1, 'C', fill=True)
    
    # Table Content
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(0, 0, 0)
    
    for _, row in valid_df.sort_values(by="Skor Risiko Numerik", ascending=False).iterrows():
        name = str(row['Nama Produk']).strip()
        if len(name) > 55: name = name[:52] + "..."
        skor = f"{row['Skor Risiko Numerik']:.2f}%"
        klas = str(row['Klasifikasi'])
        klas = tr_risk(klas)
        
        name = name.encode('latin-1', 'replace').decode('latin-1')
        
        pdf.cell(col_w[0], 10, f" {name}", 1, 0, 'L')
        pdf.cell(col_w[1], 10, skor, 1, 0, 'C')
        pdf.cell(col_w[2], 10, klas, 1, 1, 'C')

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmpdf:
        pdf.output(tmpdf.name)
        with open(tmpdf.name, 'rb') as f:
            pdf_bytes = f.read()
    os.unlink(tmpdf.name)
    
    return pdf_bytes


def generate_history_pdf_report(df_history):
    from fpdf import FPDF
    
    class PDFReport(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 16)
            self.set_text_color(15, 23, 42)
            self.cell(0, 10, t('Riwayat Analisis Lengkap - SMART NutriScan AI', 'Complete Analysis History - SMART NutriScan AI'), 0, 1, 'C')
            self.set_draw_color(200, 200, 200)
            self.line(31, 22, 266, 22) 
            self.ln(10)
            
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, t(f'Halaman {self.page_no()}', f'Page {self.page_no()}'), 0, 0, 'C')

    pdf = PDFReport(orientation='L', unit='mm', format='A4')
    pdf.set_left_margin(31)
    pdf.set_right_margin(31)
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 10, t(f"Total Riwayat: {len(df_history)} Produk", f"Total History: {len(df_history)} Products"), 0, 1)
    pdf.ln(5)
    
    col_w = [25, 20, 25, 25, 20, 25, 20, 20, 25, 30]
    total_table_width = sum(col_w)
    
    for idx, row in df_history.iterrows():
        pdf.set_font("Arial", 'B', 10)
        pdf.set_fill_color(235, 240, 248)
        
        waktu = str(row.get('Waktu Analisis', ''))[:16]
        name = str(row.get('Nama Produk', '')).strip()
        name = name.encode('latin-1', 'replace').decode('latin-1')
        
        skor_val = row.get('Skor Risiko (%)', 0)
        skor_str = t("Data belum cukup", "Insufficient data")
        skor = f"{skor_val}%" if pd.notna(skor_val) and skor_val != skor_str and skor_val != "Data belum cukup" else "-"
        klas = str(row.get('Klasifikasi', '-'))
        klas = tr_risk(klas)
        
        title_txt = t("Waktu", "Time")
        title_score = t("Skor Risiko", "Risk Score")
        title = f"{idx+1}. {name}   |   {title_txt}: {waktu}   |   {title_score}: {skor} ({klas})"
        pdf.cell(total_table_width, 8, title, 1, 1, 'L', fill=True)
        
        pdf.set_font("Arial", 'B', 8)
        pdf.set_fill_color(245, 245, 245)
        if st.session_state.lang == "ID":
            headers = ["Takaran(g)", "Energi", "Lemak Tot", "Lemak Jen", "Protein", "Karbohidrat", "Gula", "Garam", "Natrium(mg)", "N.Benzoat(mg)"]
        else:
            headers = ["Serving(g)", "Energy", "Tot Fat", "Sat Fat", "Protein", "Carbs", "Sugar", "Salt", "Sodium(mg)", "N.Benzoate(mg)"]
        
        for i in range(len(headers)):
            pdf.cell(col_w[i], 6, headers[i], 1, 0, 'C', fill=True)
        pdf.ln()
        
        pdf.set_font("Arial", '', 8)
        vals = [
            str(row.get('Takaran Saji (g/ml)', '-')),
            str(row.get('Energi (kkal)', '-')),
            str(row.get('Lemak Total (g)', '-')),
            str(row.get('Lemak Jenuh (g)', '-')),
            str(row.get('Protein (g)', '-')),
            str(row.get('Karbohidrat (g)', '-')),
            str(row.get('Gula (g)', '-')),
            str(row.get('Garam (g)', '-')),
            str(row.get('Natrium (mg)', '-')),
            str(row.get('Natrium Benzoat (mg)', '-'))
        ]
        for i in range(len(vals)):
            pdf.cell(col_w[i], 6, vals[i], 1, 0, 'C')
        pdf.ln()
        
        current_l_margin = pdf.l_margin
        
        pdf.ln(2)
        pdf.set_font("Arial", 'B', 8)
        pdf.cell(25, 5, t("Komposisi:", "Composition:"), 0, 0, 'L')
        
        x_start = pdf.get_x()
        pdf.set_left_margin(x_start)
        pdf.set_font("Arial", '', 8)
        komposisi = str(row.get('Komposisi', '-')).encode('latin-1', 'replace').decode('latin-1')
        komposisi = komposisi.replace('\n', ' ').replace('\r', '')
        pdf.multi_cell(total_table_width - 25, 5, komposisi)
        
        pdf.set_left_margin(current_l_margin)
        
        pdf.ln(1)
        pdf.set_font("Arial", 'B', 8)
        pdf.cell(25, 5, t("Rekomendasi:", "Recommendation:"), 0, 0, 'L')
        
        x_start = pdf.get_x()
        pdf.set_left_margin(x_start)
        pdf.set_font("Arial", '', 8)
        rekomendasi = str(row.get('Rekomendasi', '-')).encode('latin-1', 'replace').decode('latin-1')
        rekomendasi = rekomendasi.replace('\n', ' ').replace('\r', '')
        pdf.multi_cell(total_table_width - 25, 5, rekomendasi)
        
        pdf.set_left_margin(current_l_margin)
        pdf.ln(8)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmpdf:
        pdf.output(tmpdf.name)
        with open(tmpdf.name, 'rb') as f:
            pdf_bytes = f.read()
    os.unlink(tmpdf.name)
    
    return pdf_bytes


def render_analysis_side(analysis_result, current_signature=None):
    if not analysis_result:
        st.info(t("Hasil analisis akan muncul di sini setelah tombol analisis diklik.", "Analysis results will appear here after the analyze button is clicked."))
        return

    stored_signature = analysis_result.get("input_signature")
    if stored_signature is not None and current_signature is not None and stored_signature != current_signature:
        st.warning(t("Data input sudah berubah setelah analisis terakhir. Klik tombol analisis lagi untuk memperbarui hasil.", "Input data has changed since the last analysis. Click the analyze button again to update results."))

    if analysis_result.get("status") == "insufficient":
        st.warning(analysis_result.get("message", t("Data belum cukup untuk dianalisis.", "Insufficient data for analysis.")))
        st.info(analysis_result.get("integrity_note", t("Periksa kembali data input sebelum analisis.", "Recheck input data before analysis.")))
        return

    risk_score = float(analysis_result.get("risk_score", 0))
    xai_factors = analysis_result.get("xai_factors", {})

    render_risk_status(risk_score)
    st.markdown(t("#### Radar Kontribusi Nutrisi", "#### Nutrition Contribution Radar"))
    render_xai_radar(xai_factors)


def render_analysis_bottom(analysis_result, current_threshold):
    if not analysis_result or analysis_result.get("status") == "insufficient":
        return

    risk_score = float(analysis_result.get("risk_score", 0))
    risk_info = analysis_result.get("risk_info", classify_risk(risk_score))
    recommendation = analysis_result.get("recommendation", "")
    nutrition_data = analysis_result.get("nutrition_data", {})
    takaran_saji = analysis_result.get("takaran_saji", 0)
    
    st.markdown("---")
    render_recommendation_details(risk_info, recommendation, analysis_result.get("is_upf"), analysis_result.get("upf_flags", []))
    
    st.markdown("---")
    render_holistic_nutrition_profile(nutrition_data, takaran_saji)

    st.markdown("---")
    render_health_metrics(nutrition_data, takaran_saji, current_threshold, show_header=True)


def store_product_analysis_result(product_name, takaran_saji, nutrition_data, komposisi, store_key, input_signature=None):
    analysis_result = build_analysis_result(product_name, takaran_saji, nutrition_data, komposisi)
    analysis_result["input_signature"] = input_signature or make_analysis_signature(product_name, takaran_saji, nutrition_data, komposisi)

    st.session_state[store_key] = analysis_result

    if analysis_result.get("status") == "ok":
        risk_score = analysis_result["risk_score"]
        risk_info = analysis_result["risk_info"]
        
        if not product_name:
            product_name = t("Produk Tanpa Nama", "Unnamed Product")
        
        st.session_state.scan_history.append({
            "Waktu Analisis": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Nama Produk": product_name,
            "Skor Risiko (%)": round(risk_score, 2),
            "Klasifikasi": risk_info["label"],
            "Takaran Saji (g/ml)": takaran_saji,
            "Energi (kkal)": nutrition_data.get("energi", 0),
            "Lemak Total (g)": nutrition_data.get("lemak_total", 0),
            "Lemak Jenuh (g)": nutrition_data.get("lemak_jenuh", 0),
            "Protein (g)": nutrition_data.get("protein", 0),
            "Karbohidrat (g)": nutrition_data.get("karbohidrat", 0),
            "Gula (g)": nutrition_data.get("gula", 0),
            "Garam (g)": nutrition_data.get("garam", 0),
            "Natrium (mg)": nutrition_data.get("natrium", 0),
            "Natrium Benzoat (mg)": nutrition_data.get("natrium_benzoat", 0),
            "Komposisi": komposisi,
            "Rekomendasi": analysis_result.get("recommendation", "")
        })

    return analysis_result


def run_product_analysis(product_name, takaran_saji, nutrition_data, komposisi, current_threshold, store_key=None, input_signature=None):
    target_key = store_key or "manual_analysis_result"
    analysis_result = store_product_analysis_result(
        product_name, takaran_saji, nutrition_data, komposisi, target_key, input_signature=input_signature,
    )
    render_analysis_side(analysis_result, current_signature=analysis_result["input_signature"])
    render_analysis_bottom(analysis_result, current_threshold)
    return analysis_result


def input_form(prefix, defaults):
    name_key = f"{prefix}_name"
    saji_key = f"{prefix}_saji"
    energi_key = f"{prefix}_energi"
    lemak_key = f"{prefix}_lemak"
    jenuh_key = f"{prefix}_jenuh"
    protein_key = f"{prefix}_protein"
    karbo_key = f"{prefix}_karbo"
    gula_key = f"{prefix}_gula"
    garam_key = f"{prefix}_garam"
    natrium_key = f"{prefix}_natrium"
    benzoat_key = f"{prefix}_benzoat"
    komposisi_key = f"{prefix}_komposisi"

    set_widget_default(name_key, str(defaults.get("product_name", "")))
    set_widget_default(saji_key, float(defaults.get("takaran_saji", 100.0)))
    set_widget_default(energi_key, float(defaults.get("energi", 0.0)))
    set_widget_default(lemak_key, float(defaults.get("lemak_total", 0.0)))
    set_widget_default(jenuh_key, float(defaults.get("lemak_jenuh", 0.0)))
    set_widget_default(protein_key, float(defaults.get("protein", 0.0)))
    set_widget_default(karbo_key, float(defaults.get("karbohidrat", 0.0)))
    set_widget_default(gula_key, float(defaults.get("gula", 0.0)))
    set_widget_default(garam_key, float(defaults.get("garam", 0.0)))
    set_widget_default(natrium_key, float(defaults.get("natrium", 0.0)))
    set_widget_default(benzoat_key, float(defaults.get("natrium_benzoat", 0.0)))
    set_widget_default(komposisi_key, str(defaults.get("komposisi", "")))

    product_name = st.text_input(t("Nama Produk", "Product Name"), key=name_key)

    c0, c1, c2 = st.columns(3)
    takaran_saji = c0.number_input(t("Takaran Saji g atau ml", "Serving Size g or ml"), min_value=1.0, format="%.2f", key=saji_key)
    energi = c1.number_input(t("Energi kkal", "Energy kcal"), min_value=0.0, format="%.2f", key=energi_key)
    lemak_total = c2.number_input(t("Lemak Total g", "Total Fat g"), min_value=0.0, format="%.2f", key=lemak_key)

    c3, c4, c5 = st.columns(3)
    lemak_jenuh = c3.number_input(t("Lemak Jenuh g", "Saturated Fat g"), min_value=0.0, format="%.2f", key=jenuh_key)
    protein = c4.number_input(t("Protein g", "Protein g"), min_value=0.0, format="%.2f", key=protein_key)
    karbohidrat = c5.number_input(t("Karbohidrat g", "Carbohydrate g"), min_value=0.0, format="%.2f", key=karbo_key)

    c6, c7, c8, c9 = st.columns(4)
    gula = c6.number_input(t("Gula g", "Sugar g"), min_value=0.0, format="%.2f", key=gula_key)
    garam = c7.number_input(t("Garam g", "Salt g"), min_value=0.0, format="%.2f", key=garam_key)
    natrium = c8.number_input(t("Natrium mg", "Sodium mg"), min_value=0.0, format="%.2f", key=natrium_key)
    natrium_benzoat = c9.number_input(t("Natrium Benzoat mg", "Sodium Benzoate mg"), min_value=0.0, format="%.2f", key=benzoat_key)

    komposisi = st.text_area(t("Komposisi", "Composition"), height=120, key=komposisi_key)

    nutrition_data = build_nutrition_data(
        energi, lemak_total, lemak_jenuh, protein, karbohidrat, gula, garam, natrium, natrium_benzoat
    )

    return product_name, takaran_saji, nutrition_data, komposisi


try:
    col_hdr1, col_hdr2, col_hdr3 = st.columns([1, 4, 1])
    with col_hdr2:
        st.image("assets/Header Smart NutriScan AI.png", use_container_width=True)
except Exception:
    pass


with st.sidebar:
    lang_choice = st.radio("Language / Bahasa", ["ID", "EN"], horizontal=True)
    st.session_state.lang = lang_choice

    try:
        st.image("assets/Logo Smart NutriScan AI.png", width=150)
    except Exception:
        st.markdown("## SMART NutriScan AI")

    st.title("SMART NutriScan AI")
    st.header(t("Profil Pengguna", "User Profile"))

    col_gender, col_age = st.columns(2)
    user_gender = col_gender.selectbox(t("Gender", "Gender"), [t("Pria", "Male"), t("Wanita", "Female")])
    user_age = col_age.number_input(t("Usia", "Age"), min_value=1, max_value=120, value=25)

    col_weight, col_height = st.columns(2)
    user_weight = col_weight.number_input(t("Berat kg", "Weight kg"), min_value=10.0, max_value=300.0, value=65.0)
    user_height = col_height.number_input(t("Tinggi cm", "Height cm"), min_value=50.0, max_value=250.0, value=165.0)

    act_list = [t("Sedentary", "Sedentary"), t("Ringan", "Light"), t("Sedang", "Moderate"), t("Aktif", "Active"), t("Sangat Aktif", "Very Active")]
    user_activity = st.selectbox(t("Aktivitas", "Activity"), act_list)
    
    med_list = [t("Tidak Ada", "None"), t("Penderita Hipertensi", "Hypertension"), t("Risiko Penyakit Ginjal", "Kidney Disease Risk"), t("Anak anak", "Children")]
    kondisi_medis = st.selectbox(t("Kondisi Khusus", "Medical Condition"), med_list)

    current_threshold = hitung_tdee_dinamis(user_gender, user_age, user_weight, user_height, user_activity)
    
    if kondisi_medis in ["Penderita Hipertensi", "Hypertension"]:
        current_threshold["natrium"] = 1200
    elif kondisi_medis in ["Risiko Penyakit Ginjal", "Kidney Disease Risk"]:
        current_threshold["natrium"] = 1000
        current_threshold["kalori"] *= 0.9
    elif kondisi_medis in ["Anak anak", "Children"]:
        current_threshold["gula"] = 25
        current_threshold["natrium"] = 1500

    with st.expander(t("Lihat batas harian", "View daily limits")):
        st.write(f"{t('Kalori', 'Calories')}: {current_threshold['kalori']:.2f} kcal")
        st.write(f"{t('Gula', 'Sugar')}: {current_threshold['gula']:.2f} g")
        st.write(f"{t('Lemak jenuh', 'Saturated Fat')}: {current_threshold['lemak_jenuh']:.2f} g")
        st.write(f"{t('Natrium', 'Sodium')}: {current_threshold['natrium']:.2f} mg")

    feature_options = [
        t("Analisis Produk Tunggal", "Single Product Analysis"),
        "Scan from Image",
        t("Analisis Batch Excel", "Batch Excel Analysis"),
        t("Perbandingan Produk", "Product Comparison"),
        t("Simulasi Konsumsi Produk", "Consumption Simulation"),
        t("Riwayat Analisis", "Analysis History"),
        t("Edukasi Gizi", "Nutrition Education"),
    ]
    app_mode = st.radio(t("Pilih Fitur", "Select Feature"), feature_options)


st.title("SMART NutriScan AI")
st.caption(t(
    "Analisis produk pangan berbasis OCR, machine learning, aturan gizi terkalibrasi, dan konfirmasi data manual.",
    "Food product analysis based on OCR, machine learning, calibrated nutrition rules, and manual data confirmation."
))

model_ready = all([feat_model, lgbm_model, w2v_model, scaler])
if model_ready:
    st.success(t("Model utama berhasil dimuat. Skor tetap dijaga oleh aturan gizi agar klasifikasi konsisten.", "Main model loaded successfully. Score is maintained by nutrition rules to ensure classification consistency."))
else:
    st.warning(t("Sebagian model utama belum terbaca. Aplikasi tetap berjalan dengan analisis gizi terkalibrasi.", "Some main models are not readable. App is still running with calibrated nutritional analysis."))


if app_mode in ["Analisis Produk Tunggal", "Single Product Analysis"]:
    st.header(t("Analisis Produk Tunggal", "Single Product Analysis"))

    manual_input_col, manual_result_col = st.columns([1.15, 1], gap="large")

    with manual_input_col:
        st.subheader(t("Input Informasi Produk", "Input Product Information"))
        
        if "preset_selector" not in st.session_state:
            st.session_state.preset_selector = list(EXAMPLE_PRESETS.keys())[0]

        preset_options = list(EXAMPLE_PRESETS.keys())
        st.selectbox(
            t("Pilih contoh uji atau isi manual", "Select example or fill manually"), 
            preset_options, 
            key="preset_selector",
            format_func=lambda x: t(x, EN_PRESET_NAMES.get(x, x)),
            on_change=apply_manual_preset
        )
        
        product_name, takaran_saji, nutrition_data, komposisi = input_form("manual", EXAMPLE_PRESETS["Kosong"])
        manual_signature = make_analysis_signature(product_name, takaran_saji, nutrition_data, komposisi)

        if st.button(t("Analisis AI dan Gizi", "Analyze AI and Nutrition"), type="primary"):
            store_product_analysis_result(
                product_name,
                takaran_saji,
                nutrition_data,
                komposisi,
                store_key="manual_analysis_result",
                input_signature=manual_signature,
            )
            st.success(t("Analisis berhasil diperbarui. Hasil ditampilkan di panel kanan.", "Analysis successfully updated. Results are displayed on the right panel."))

    with manual_result_col:
        st.markdown(t("### Profil & Analisis Produk Dasar", "### Basic Product Profile & Analysis"))
        st.caption(t("Ringkasan prediksi risiko dan kontribusi nutrisi utama.", "Summary of predicted risk and main nutritional contributions."))
        render_analysis_side(st.session_state.manual_analysis_result, current_signature=manual_signature)

    render_analysis_bottom(st.session_state.manual_analysis_result, current_threshold)


elif app_mode == "Scan from Image":
    st.header(t("Scan Produk Otomatis", "Automatic Product Scan"))
    st.info(t(
        "Ambil foto dekat, lurus, tidak blur, dan pastikan label memenuhi sebagian besar area gambar. Setelah OCR selesai, koreksi data sebelum analisis.",
        "Take a close, straight, non-blurry photo, and ensure the label covers most of the image area. After OCR completes, correct the data before analysis."
    ))

    if st.button(t("Reset Hasil OCR", "Reset OCR Results")):
        st.session_state.ocr_data = init_parsed_data()
        clear_ocr_analysis_result()
        bump_ocr_form_version()
        st.success(t("Hasil OCR dan analisis terakhir sudah dikosongkan.", "Last OCR and analysis results have been cleared."))

    col_scan1, col_scan2 = st.columns(2)

    with col_scan1:
        st.subheader(t("Scan 1: Informasi Nilai Gizi", "Scan 1: Nutritional Value Info"))
        input_type_1 = st.radio(t("Metode input nilai gizi", "Nutrition value input method"), [t("Upload File", "Upload File"), t("Kamera Langsung", "Live Camera")], key="input_gizi")
        
        if input_type_1 in ["Upload File"]:
            img_file_1 = st.file_uploader(t("Upload foto nilai gizi", "Upload nutrition value photo"), type=["jpg", "jpeg", "png"], key="upload_gizi") 
        else:
            img_file_1 = st.camera_input(t("Foto nilai gizi", "Nutrition photo"), key="camera_gizi")

        if img_file_1 is not None:
            try:
                image_1_original = Image.open(img_file_1)
                image_1_display = standardize_image_size(image_1_original, target_ratio=4/3)
                safe_image(image_1_display, caption=t("Gambar nilai gizi", "Nutrition image"), width=350)

                if st.button(t("Proses OCR Nilai Gizi", "Process Nutrition OCR"), key="btn_ocr_gizi"):
                    with st.spinner(t("Mempersiapkan OCR dan membaca nilai gizi secara bertahap...", "Preparing OCR and reading nutritional values...")):
                        reader, reader_error = get_ocr_reader_safely()
                        if reader_error:
                            scan_result_1, ocr_error_1 = None, reader_error
                        else:
                            scan_result_1, ocr_error_1 = run_ocr_safely(reader, image_1_original, mode="nutrition")

                    if ocr_error_1:
                        st.error(t("OCR nilai gizi gagal diproses. Aplikasi tidak dihentikan. Silakan input manual atau coba foto yang lebih jelas.", "Nutrition OCR failed. App is not stopped. Please input manually or try a clearer photo."))
                        with st.expander(t("Detail error OCR nilai gizi", "Nutrition OCR error details")):
                            st.code(ocr_error_1)
                    else:
                        parsed_gizi = scan_result_1["parsed"]
                        changed = False
                        for key, value in parsed_gizi.items():
                            if key in st.session_state.ocr_data:
                                if value not in [0, 0.0, "Tidak terdeteksi.", "Produk Tanpa Nama", ""]:
                                    sync_ocr_value_to_form(key, value)
                                    changed = True
                        if changed:
                            clear_ocr_analysis_result()
                            bump_ocr_form_version()

                        st.success(t("Nilai gizi berhasil diproses. Angka satuan g yang terbaca sebagai 9 sudah dikoreksi sebelum masuk form. Periksa lagi sebelum analisis.", "Nutritional values successfully processed. Check again before analysis."))
                        render_ocr_result_debug(scan_result_1, t("nilai gizi", "nutrition values"))
            except Exception as exc:
                st.error(t("Gambar nilai gizi tidak bisa dibaca. Coba upload ulang dalam format JPG atau PNG.", "Nutrition image cannot be read. Try re-uploading in JPG or PNG format."))
                with st.expander(t("Detail error gambar nilai gizi", "Nutrition image error details")):
                    st.code(str(exc))

    with col_scan2:
        st.subheader(t("Scan 2: Komposisi Produk", "Scan 2: Product Composition"))
        input_type_2 = st.radio(t("Metode input komposisi", "Composition input method"), [t("Upload File", "Upload File"), t("Kamera Langsung", "Live Camera")], key="input_komposisi")
        
        if input_type_2 in ["Upload File"]:
            img_file_2 = st.file_uploader(t("Upload foto komposisi", "Upload composition photo"), type=["jpg", "jpeg", "png"], key="upload_komposisi") 
        else:
            img_file_2 = st.camera_input(t("Foto komposisi", "Composition photo"), key="camera_komposisi")

        if img_file_2 is not None:
            try:
                image_2_original = Image.open(img_file_2)
                image_2_display = standardize_image_size(image_2_original, target_ratio=4/3)
                safe_image(image_2_display, caption=t("Gambar komposisi", "Composition image"), width=350)

                if st.button(t("Proses OCR Komposisi", "Process Composition OCR"), key="btn_ocr_komposisi"):
                    with st.spinner(t("Mempersiapkan OCR dan membaca komposisi secara bertahap...", "Preparing OCR and reading composition...")):
                        reader, reader_error = get_ocr_reader_safely()
                        if reader_error:
                            scan_result_2, ocr_error_2 = None, reader_error
                        else:
                            scan_result_2, ocr_error_2 = run_ocr_safely(reader, image_2_original, mode="composition")

                    if ocr_error_2:
                        st.error(t("OCR komposisi gagal diproses. Aplikasi tidak dihentikan. Silakan input manual atau coba foto yang lebih jelas.", "Composition OCR failed. App is not stopped. Please input manually or try a clearer photo."))
                        with st.expander(t("Detail error OCR komposisi", "Composition OCR error details")):
                            st.code(ocr_error_2)
                    else:
                        parsed_komposisi = scan_result_2["parsed"].get("komposisi", "Tidak terdeteksi.")
                        if parsed_komposisi != "Tidak terdeteksi.":
                            sync_ocr_value_to_form("komposisi", parsed_komposisi)
                            clear_ocr_analysis_result()
                            bump_ocr_form_version()

                        st.success(t("Komposisi berhasil diproses dari satu variasi OCR terbaik agar tidak berulang. Periksa lagi sebelum analisis.", "Composition successfully processed from the best OCR variant. Check again before analysis."))
                        render_ocr_result_debug(scan_result_2, t("komposisi", "composition"))
            except Exception as exc:
                st.error(t("Gambar komposisi tidak bisa dibaca. Coba upload ulang dalam format JPG atau PNG.", "Composition image cannot be read. Try re-uploading in JPG or PNG format."))
                with st.expander(t("Detail error gambar komposisi", "Composition image error details")):
                    st.code(str(exc))

    st.markdown("---")
    input_col, result_col = st.columns([1.15, 1], gap="large")

    with input_col:
        st.subheader(t("Konfirmasi Data Input (Hasil OCR)", "Confirm Input Data (OCR Results)"))
        st.warning(t("Jangan langsung percaya OCR mentah. Koreksi angka dan komposisi sebelum menjalankan rekomendasi.", "Do not blindly trust raw OCR. Correct numbers and composition before running recommendations."))

        ocr_prefix = f"ocr_{st.session_state.ocr_form_version}"
        product_name, takaran_saji, nutrition_data, komposisi = input_form(ocr_prefix, st.session_state.ocr_data)
        ocr_signature = make_analysis_signature(product_name, takaran_saji, nutrition_data, komposisi)

        if st.button(t("Analisis dari Data Hasil OCR", "Analyze from OCR Data"), type="primary"):
            store_product_analysis_result(
                product_name,
                takaran_saji,
                nutrition_data,
                komposisi,
                store_key="ocr_analysis_result",
                input_signature=ocr_signature,
            )
            st.success(t("Analisis berhasil diperbarui. Hasil ditampilkan di panel kanan.", "Analysis successfully updated. Results are displayed on the right panel."))

    with result_col:
        st.markdown(t("### Profil & Analisis Produk Dasar", "### Basic Product Profile & Analysis"))
        st.caption(t("Ringkasan prediksi risiko dan kontribusi nutrisi utama.", "Summary of predicted risk and main nutritional contributions."))
        render_analysis_side(st.session_state.ocr_analysis_result, current_signature=ocr_signature)
        
    render_analysis_bottom(st.session_state.ocr_analysis_result, current_threshold)


elif app_mode in ["Analisis Batch Excel", "Batch Excel Analysis"]:
    st.header(t("Analisis Batch Excel", "Batch Excel Analysis"))
    st.write(t(
        "Upload file Excel dengan kolom Nama Produk, Energi, Lemak, Lemak Jenuh, Karbohidrat, Gula, Protein, Garam, Natrium, Natrium Benzoat, dan Komposisi jika tersedia.",
        "Upload an Excel file with columns Product Name, Energy, Fat, Saturated Fat, Carbohydrate, Sugar, Protein, Salt, Sodium, Sodium Benzoate, and Composition if available."
    ))

    uploaded_file = st.file_uploader(t("Upload Excel", "Upload Excel"), type=["xlsx"])
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        
        numeric_cols_to_fix = [
            "Energi", "Energy", "Lemak", "Fat", "Lemak Total", "Total Fat", "Lemak Jenuh", "Saturated Fat", 
            "Karbohidrat", "Carbohydrate", "Gula", "Sugar", "Protein", "Garam", "Salt", 
            "Natrium", "Sodium", "Natrium Benzoat", "Sodium Benzoate", "Takaran Saji", "Serving Size", "Takaran"
        ]
        
        def fix_excel_date_bug(val):
            if pd.isna(val):
                return 0.0
            if isinstance(val, datetime) or type(val).__name__ == 'Timestamp':
                return float(f"{val.day}.{val.month}")
            if isinstance(val, str):
                val = val.strip()
                if not val:
                    return 0.0
                if re.match(r'^\d{4}-\d{2}-\d{2}', val):
                    try:
                        dt = pd.to_datetime(val)
                        return float(f"{dt.day}.{dt.month}")
                    except Exception:
                        pass
                val = val.replace(',', '.')
                try:
                    return float(val)
                except ValueError:
                    return 0.0
            try:
                return float(val)
            except Exception:
                return 0.0

        for col in numeric_cols_to_fix:
            if col in df.columns:
                df[col] = df[col].apply(fix_excel_date_bug)
        
        st.dataframe(df, use_container_width=True)
        
        if st.button(t("Mulai Analisis Batch", "Start Batch Analysis"), type="primary"):
            df_clean = preprocess_batch_excel_data(df)
            results = []
            total_rows = len(df_clean)
            
            progress_bar = st.progress(0)
            
            counter = 0
            for idx, row in df_clean.iterrows():
                nutrition_data = {
                    "energi": row.get("Energi", row.get("Energy", 0)),
                    "lemak_total": row.get("Lemak", row.get("Lemak Total", row.get("Fat", row.get("Total Fat", 0)))),
                    "lemak_jenuh": row.get("Lemak Jenuh", row.get("Saturated Fat", 0)),
                    "protein": row.get("Protein", 0),
                    "karbohidrat": row.get("Karbohidrat", row.get("Carbohydrate", 0)),
                    "gula": row.get("Gula", row.get("Sugar", 0)),
                    "garam": row.get("Garam", row.get("Salt", 0)),
                    "natrium": row.get("Natrium", row.get("Sodium", 0)),
                    "natrium_benzoat": row.get("Natrium Benzoat", row.get("Sodium Benzoate", 0)),
                }
                komposisi = row.get("Komposisi", row.get("Composition", ""))
                
                takaran_saji = float(row.get("Takaran Saji", row.get("Takaran", row.get("Serving Size", 100.0))))
                product_name = str(row.get("Nama Produk", row.get("Produk", row.get("Product Name", f"Product {idx+1}"))))

                if not has_sufficient_input(nutrition_data):
                    results.append({
                        "Nama Produk": product_name,
                        "Skor Risiko": t("Data belum cukup", "Insufficient data"),
                        "Klasifikasi": t("Belum dianalisis", "Not analyzed"),
                        "Rekomendasi": t("Isi nilai gizi yang valid sebelum analisis.", "Fill in valid nutrition values before analysis."),
                        "nutrition_data": nutrition_data,
                        "takaran_saji": takaran_saji
                    })
                else:
                    risk_score, _, recommendation = analyze_product_fully(
                        nutrition_data,
                        komposisi,
                        feat_model,
                        lgbm_model,
                        w2v_model,
                        scaler,
                    )
                    risk_info = classify_risk(risk_score)
                    results.append({
                        "Nama Produk": product_name,
                        "Skor Risiko": round(risk_score, 2),
                        "Klasifikasi": risk_info["label"],
                        "Rekomendasi": recommendation,
                        "nutrition_data": nutrition_data,
                        "takaran_saji": takaran_saji
                    })

                    # --- SIMPAN KE HISTORY ---
                    st.session_state.scan_history.append({
                        "Waktu Analisis": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Nama Produk": product_name,
                        "Skor Risiko (%)": round(risk_score, 2),
                        "Klasifikasi": risk_info["label"],
                        "Takaran Saji (g/ml)": takaran_saji,
                        "Energi (kkal)": nutrition_data.get("energi", 0),
                        "Lemak Total (g)": nutrition_data.get("lemak_total", 0),
                        "Lemak Jenuh (g)": nutrition_data.get("lemak_jenuh", 0),
                        "Protein (g)": nutrition_data.get("protein", 0),
                        "Karbohidrat (g)": nutrition_data.get("karbohidrat", 0),
                        "Gula (g)": nutrition_data.get("gula", 0),
                        "Garam (g)": nutrition_data.get("garam", 0),
                        "Natrium (mg)": nutrition_data.get("natrium", 0),
                        "Natrium Benzoat (mg)": nutrition_data.get("natrium_benzoat", 0),
                        "Komposisi": komposisi,
                        "Rekomendasi": recommendation
                    })
                
                counter += 1
                progress_bar.progress(counter / total_rows)
            
            st.session_state.batch_result_df = pd.DataFrame(results)
            st.session_state.batch_total_rows = total_rows
            
        if st.session_state.batch_result_df is not None:
            st.success(t(f"Analisis batch selesai untuk {st.session_state.batch_total_rows} produk!", f"Batch analysis completed for {st.session_state.batch_total_rows} products!"))
            
            st.header(t("Hasil Analisis Batch", "Batch Analysis Results"))

            st.markdown(t("### 🤖 Ringkasan Insight Otomatis", "### 🤖 Automated Insight Summary"))
            insight_text = generate_batch_insights(st.session_state.batch_result_df)
            st.info(insight_text)
            st.markdown("---")
            
            display_cols = ["Nama Produk", "Skor Risiko", "Klasifikasi", "Rekomendasi"]
            
            # Translate DataFrame display conditionally
            df_disp = st.session_state.batch_result_df[display_cols].copy()
            if st.session_state.lang == "EN":
                df_disp.rename(columns={
                    "Nama Produk": "Product Name",
                    "Skor Risiko": "Risk Score",
                    "Klasifikasi": "Classification",
                    "Rekomendasi": "Recommendation"
                }, inplace=True)
                df_disp["Classification"] = df_disp["Classification"].apply(tr_risk)
                st.dataframe(df_disp, use_container_width=True)
            else:
                st.dataframe(df_disp, use_container_width=True)
            
            try:
                st.markdown("---")
                st.markdown(t("### 2. Grafik Distribusi Risiko", "### 2. Risk Distribution Chart"))

                df_results = st.session_state.batch_result_df.copy()
                
                df_results["Skor Risiko Numerik"] = pd.to_numeric(df_results["Skor Risiko"], errors='coerce')
                valid_df = df_results.dropna(subset=["Skor Risiko Numerik"])

                fig_pie = None

                if not valid_df.empty:
                    classification_colors = {"Aman": "#2ECC71", "Sedang": "#F39C12", "Tinggi": "#E74C3C"}

                    # === PIE CHART ===
                    st.markdown(t("#### Proporsi Klasifikasi Produk", "#### Product Classification Proportion"))
                    
                    pie_data = valid_df["Klasifikasi"].value_counts().reset_index()
                    pie_data.columns = ["Klasifikasi", "Jumlah"]
                    pie_colors = [classification_colors.get(c, "#95A5A6") for c in pie_data["Klasifikasi"]]
                    
                    # Translate labels for UI
                    pie_labels = [tr_risk(c) for c in pie_data["Klasifikasi"]]

                    fig_pie = go.Figure(data=[go.Pie(
                        labels=pie_labels, 
                        values=pie_data["Jumlah"], 
                        hole=0.4,
                        marker=dict(colors=pie_colors),
                        hoverinfo='label+percent+value',
                        textinfo='percent+label',
                        textfont_size=15
                    )])
                    
                    fig_pie.update_layout(
                        showlegend=True, 
                        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                        margin=dict(t=30, b=40, l=20, r=20), 
                        height=450
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                    st.markdown("<hr style='border:1px dashed #E2E8F0; margin: 30px 0;'>", unsafe_allow_html=True)

                    # === BAR CHART HTML ===
                    st.markdown(t("#### Peringkat Skor Risiko Produk", "#### Product Risk Score Ranking"))
                    
                    bar_data = valid_df.sort_values(by="Skor Risiko Numerik", ascending=False)
                    
                    modern_palette = [
                        "#F59E0B", "#3B82F6", "#8B5CF6", "#10B981", "#EF4444", 
                        "#06B6D4", "#F97316", "#EC4899", "#84CC16", "#14B8A6",
                        "#6366F1", "#F43F5E", "#0EA5E9", "#10B981", "#8B5CF6"
                    ]

                    html_bars = "<div style='margin-top: 16px;'>"
                    for i, (_, row_data) in enumerate(bar_data.iterrows()):
                        prod_name = row_data["Nama Produk"]
                        score = float(row_data["Skor Risiko Numerik"])
                        color = modern_palette[i % len(modern_palette)]
                        html_bars += f"<div style='margin-bottom: 22px; font-family: \"Inter\", \"Segoe UI\", sans-serif;'><div style='display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 8px;'><span style='font-weight: 600; font-size: 14.5px; color: #0F172A;'>{prod_name}</span><span style='font-weight: 700; font-size: 14px; color: #334155;'>{score:.1f}%</span></div><div style='width: 100%; background-color: #E2E8F0; border-radius: 8px; height: 14px; box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);'><div style='width: {min(score, 100)}%; background-color: {color}; height: 100%; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.1); transition: width 0.8s ease-out;'></div></div></div>"
                    html_bars += "</div>"
                    
                    st.markdown(html_bars, unsafe_allow_html=True)
                    
                else:
                    st.info(t("Tidak ada data valid yang bisa divisualisasikan dalam grafik.", "No valid data to visualize in the chart."))
                    
                st.markdown("---")
                st.markdown(t("### 🎯 Detail Pemenuhan Angka Kecukupan Gizi Harian per Produk", "### 🎯 Detailed Daily Nutrition Adequacy per Product"))
                
                for idx, row in valid_df.iterrows():
                    prod_name = row.get("Nama Produk", t("Produk", "Product"))
                    klasifikasi = tr_risk(row.get("Klasifikasi", "-"))
                    skor_num = row.get("Skor Risiko Numerik", 0)
                    
                    txt_klas = t("Klasifikasi", "Classification")
                    txt_skor = t("Skor", "Score")
                    with st.expander(f"📦 {prod_name} — {txt_klas}: {klasifikasi} ({txt_skor}: {skor_num:.1f}%)"):
                        if 'nutrition_data' in row and isinstance(row['nutrition_data'], dict):
                            nut_data = row['nutrition_data']
                        else:
                            nut_data = {"gula": 0, "natrium": 0, "lemak_jenuh": 0}
                            
                        t_saji = row['takaran_saji'] if 'takaran_saji' in row else 100.0
                        render_health_metrics(nut_data, t_saji, current_threshold, show_header=False)
                
            except Exception as e:
                st.error(t(f"Terjadi masalah saat merender visualisasi batch: {e}", f"Problem occurred when rendering batch visualization: {e}"))

            st.markdown("---")
            st.markdown(t("### 📥 Export Hasil Analisis", "### 📥 Export Analysis Results"))

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                # To excel formatting
                df_to_excel = st.session_state.batch_result_df[display_cols].copy()
                if st.session_state.lang == "EN":
                    df_to_excel.rename(columns={
                        "Nama Produk": "Product Name",
                        "Skor Risiko": "Risk Score",
                        "Klasifikasi": "Classification",
                        "Rekomendasi": "Recommendation"
                    }, inplace=True)
                    df_to_excel["Classification"] = df_to_excel["Classification"].apply(tr_risk)
                df_to_excel.to_excel(writer, index=False, sheet_name=t("Hasil Analisis", "Analysis Result"))

            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                st.download_button(
                    label=t("📊 Download Hasil Excel", "📊 Download Excel Results"), 
                    data=output.getvalue(), 
                    file_name="hasil_analisis_nutriscan.xlsx",
                    use_container_width=True
                )
                
            with col_dl2:
                try:
                    from fpdf import FPDF
                    pdf_ready = True
                except ImportError:
                    pdf_ready = False
                    
                if pdf_ready:
                    with st.spinner(t("Menyiapkan dokumen PDF Profesional...", "Preparing Professional PDF Document...")):
                        pdf_data = generate_pdf_report(df_results, insight_text, fig_pie)
                        st.download_button(
                            label=t("📑 Download PDF Report", "📑 Download PDF Report"), 
                            data=pdf_data, 
                            file_name="Laporan_Analisis_NutriScan.pdf", 
                            mime="application/pdf",
                            use_container_width=True
                        )
                else:
                    st.warning(t("⚠️ Modul 'fpdf' belum terinstall. Jalankan `pip install fpdf kaleido` di terminal agar tombol Export PDF muncul.", "⚠️ Module 'fpdf' not installed. Run `pip install fpdf kaleido` to show Export PDF button."))
            
    else:
        st.session_state.batch_result_df = None


elif app_mode in ["Perbandingan Produk", "Product Comparison"]:
    st.header(t("Perbandingan Produk (Food Comparison Mode)", "Product Comparison (Food Comparison Mode)"))
    st.info(t(
        "Bandingkan metrik AI (Skor Risiko) dan metrik BI (Kepadatan Energi) dari dua produk sekaligus. Anda dapat menggunakan preset atau OCR untuk memindai label masing-masing produk.",
        "Compare AI metrics (Risk Score) and BI metrics (Energy Density) of two products simultaneously. Use presets or OCR to scan labels of each product."
    ))

    if "comp_a_data" not in st.session_state:
        st.session_state.comp_a_data = init_parsed_data()
    if "comp_b_data" not in st.session_state:
        st.session_state.comp_b_data = init_parsed_data()
    if "comp_a_ver" not in st.session_state:
        st.session_state.comp_a_ver = 0
    if "comp_b_ver" not in st.session_state:
        st.session_state.comp_b_ver = 0

    def apply_comp_preset_a():
        st.session_state.comp_a_data = dict(EXAMPLE_PRESETS[st.session_state.preset_comp_a])
        st.session_state.comp_a_ver += 1

    def apply_comp_preset_b():
        st.session_state.comp_b_data = dict(EXAMPLE_PRESETS[st.session_state.preset_comp_b])
        st.session_state.comp_b_ver += 1

    colA, colB = st.columns(2, gap="large")

    method_options = [t("Pilih Contoh Produk", "Choose Example"), t("Scan Label (OCR)", "Scan Label (OCR)"), t("Input Manual", "Manual Input")]

    with colA:
        st.markdown(t("### 📦 Produk A", "### 📦 Product A"))
        method_a = st.radio(t("Metode Input Produk A:", "Product A Input Method:"), method_options, horizontal=True, key="method_a")
        
        if method_a in ["Pilih Contoh Produk", "Choose Example"]:
            st.selectbox(t("Contoh Uji Produk A", "Example Test Product A"), list(EXAMPLE_PRESETS.keys()), key="preset_comp_a", format_func=lambda x: t(x, EN_PRESET_NAMES.get(x, x)), on_change=apply_comp_preset_a)
        elif method_a in ["Scan Label (OCR)", "Scan Label (OCR)"]:
            col_scan_gizi_a, col_scan_komp_a = st.columns(2)
            with col_scan_gizi_a:
                st.markdown(t("**1. Scan Nilai Gizi**", "**1. Scan Nutrition**"))
                type_a_gizi = st.radio("S. Gizi A", [t("Upload", "Upload"), t("Kamera", "Camera")], key="type_a_gizi", horizontal=True, label_visibility="collapsed")
                file_a_gizi = st.file_uploader(t("Foto Gizi A", "Nutrition Photo A"), type=["jpg", "jpeg", "png"], key="file_a_gizi") if type_a_gizi in ["Upload"] else st.camera_input(t("Kamera Gizi A", "Nutrition Camera A"), key="cam_a_gizi")
                if file_a_gizi:
                    img_a_gizi = Image.open(file_a_gizi)
                    safe_image(standardize_image_size(img_a_gizi), caption=t("Gizi A", "Nutrition A"), width=250)
                    if st.button("🔍 OCR Gizi A", key="btn_ocr_a_gizi"):
                        with st.spinner(t("Membaca...", "Reading...")):
                            reader, err = get_ocr_reader_safely()
                            if err: st.error(err)
                            else:
                                res, err2 = run_ocr_safely(reader, img_a_gizi, "nutrition")
                                if err2: st.error(err2)
                                else:
                                    for k, v in res["parsed"].items():
                                        if v not in [0, 0.0, "Tidak terdeteksi.", "", "Produk Tanpa Nama"]:
                                            st.session_state.comp_a_data[k] = v
                                    st.session_state.comp_a_ver += 1
                                    st.success(t("Tersimpan!", "Saved!"))

            with col_scan_komp_a:
                st.markdown(t("**2. Scan Komposisi**", "**2. Scan Composition**"))
                type_a_komp = st.radio("S. Komp A", [t("Upload", "Upload"), t("Kamera", "Camera")], key="type_a_komp", horizontal=True, label_visibility="collapsed")
                file_a_komp = st.file_uploader(t("Foto Komp A", "Composition Photo A"), type=["jpg", "jpeg", "png"], key="file_a_komp") if type_a_komp in ["Upload"] else st.camera_input(t("Kamera Komp A", "Composition Camera A"), key="cam_a_komp")
                if file_a_komp:
                    img_a_komp = Image.open(file_a_komp)
                    safe_image(standardize_image_size(img_a_komp), caption=t("Komp A", "Comp A"), width=250)
                    if st.button("🔍 OCR Komposisi A", key="btn_ocr_a_komp"):
                        with st.spinner(t("Membaca...", "Reading...")):
                            reader, err = get_ocr_reader_safely()
                            if err: st.error(err)
                            else:
                                res, err2 = run_ocr_safely(reader, img_a_komp, "composition")
                                if err2: st.error(err2)
                                else:
                                    val = res["parsed"].get("komposisi", "")
                                    if val and val != "Tidak terdeteksi.":
                                        st.session_state.comp_a_data["komposisi"] = val
                                    st.session_state.comp_a_ver += 1
                                    st.success(t("Tersimpan!", "Saved!"))

        st.markdown(t("#### Form Data Produk A", "#### Product A Data Form"))
        prod_name_a, saji_a, nut_a, kompo_a = input_form(f"comp_a_{st.session_state.comp_a_ver}", st.session_state.comp_a_data)

    with colB:
        st.markdown(t("### 📦 Produk B", "### 📦 Product B"))
        method_b = st.radio(t("Metode Input Produk B:", "Product B Input Method:"), method_options, horizontal=True, key="method_b")
        
        if method_b in ["Pilih Contoh Produk", "Choose Example"]:
            st.selectbox(t("Contoh Uji Produk B", "Example Test Product B"), list(EXAMPLE_PRESETS.keys()), key="preset_comp_b", format_func=lambda x: t(x, EN_PRESET_NAMES.get(x, x)), on_change=apply_comp_preset_b)
        elif method_b in ["Scan Label (OCR)", "Scan Label (OCR)"]:
            col_scan_gizi_b, col_scan_komp_b = st.columns(2)
            with col_scan_gizi_b:
                st.markdown(t("**1. Scan Nilai Gizi**", "**1. Scan Nutrition**"))
                type_b_gizi = st.radio("S. Gizi B", [t("Upload", "Upload"), t("Kamera", "Camera")], key="type_b_gizi", horizontal=True, label_visibility="collapsed")
                file_b_gizi = st.file_uploader(t("Foto Gizi B", "Nutrition Photo B"), type=["jpg", "jpeg", "png"], key="file_b_gizi") if type_b_gizi in ["Upload"] else st.camera_input(t("Kamera Gizi B", "Nutrition Camera B"), key="cam_b_gizi")
                if file_b_gizi:
                    img_b_gizi = Image.open(file_b_gizi)
                    safe_image(standardize_image_size(img_b_gizi), caption=t("Gizi B", "Nutrition B"), width=250)
                    if st.button("🔍 OCR Gizi B", key="btn_ocr_b_gizi"):
                        with st.spinner(t("Membaca...", "Reading...")):
                            reader, err = get_ocr_reader_safely()
                            if err: st.error(err)
                            else:
                                res, err2 = run_ocr_safely(reader, img_b_gizi, "nutrition")
                                if err2: st.error(err2)
                                else:
                                    for k, v in res["parsed"].items():
                                        if v not in [0, 0.0, "Tidak terdeteksi.", "", "Produk Tanpa Nama"]:
                                            st.session_state.comp_b_data[k] = v
                                    st.session_state.comp_b_ver += 1
                                    st.success(t("Tersimpan!", "Saved!"))

            with col_scan_komp_b:
                st.markdown(t("**2. Scan Komposisi**", "**2. Scan Composition**"))
                type_b_komp = st.radio("S. Komp B", [t("Upload", "Upload"), t("Kamera", "Camera")], key="type_b_komp", horizontal=True, label_visibility="collapsed")
                file_b_komp = st.file_uploader(t("Foto Komp B", "Composition Photo B"), type=["jpg", "jpeg", "png"], key="file_b_komp") if type_b_komp in ["Upload"] else st.camera_input(t("Kamera Komp B", "Composition Camera B"), key="cam_b_komp")
                if file_b_komp:
                    img_b_komp = Image.open(file_b_komp)
                    safe_image(standardize_image_size(img_b_komp), caption=t("Komp B", "Comp B"), width=250)
                    if st.button("🔍 OCR Komposisi B", key="btn_ocr_b_komp"):
                        with st.spinner(t("Membaca...", "Reading...")):
                            reader, err = get_ocr_reader_safely()
                            if err: st.error(err)
                            else:
                                res, err2 = run_ocr_safely(reader, img_b_komp, "composition")
                                if err2: st.error(err2)
                                else:
                                    val = res["parsed"].get("komposisi", "")
                                    if val and val != "Tidak terdeteksi.":
                                        st.session_state.comp_b_data["komposisi"] = val
                                    st.session_state.comp_b_ver += 1
                                    st.success(t("Tersimpan!", "Saved!"))

        st.markdown(t("#### Form Data Produk B", "#### Product B Data Form"))
        prod_name_b, saji_b, nut_b, kompo_b = input_form(f"comp_b_{st.session_state.comp_b_ver}", st.session_state.comp_b_data)

    st.markdown("---")
    
    if st.button(t("⚖️ Bandingkan Kedua Produk", "⚖️ Compare Both Products"), type="primary", use_container_width=True):
        res_a = build_analysis_result(prod_name_a, saji_a, nut_a, kompo_a)
        res_b = build_analysis_result(prod_name_b, saji_b, nut_b, kompo_b)

        if res_a.get("status") == "ok" and res_b.get("status") == "ok":
            st.markdown(t("### 🏆 Kesimpulan Perbandingan AI", "### 🏆 AI Comparison Conclusion"))
            score_a = res_a["risk_score"]
            score_b = res_b["risk_score"]
            diff = abs(score_a - score_b)

            prod_a_display = res_a['product_name'] or t('Produk A', 'Product A')
            prod_b_display = res_b['product_name'] or t('Produk B', 'Product B')

            if score_a < score_b:
                st.success(t(f"Berdasarkan analisis nutrisi, **{prod_a_display}** adalah pilihan yang lebih baik. Skor risikonya **{diff:.2f}% lebih rendah** dibandingkan {prod_b_display}.", f"Based on nutritional analysis, **{prod_a_display}** is the better choice. Its risk score is **{diff:.2f}% lower** than {prod_b_display}."))
            elif score_b < score_a:
                st.success(t(f"Berdasarkan analisis nutrisi, **{prod_b_display}** adalah pilihan yang lebih baik. Skor risikonya **{diff:.2f}% lebih rendah** dibandingkan {prod_a_display}.", f"Based on nutritional analysis, **{prod_b_display}** is the better choice. Its risk score is **{diff:.2f}% lower** than {prod_a_display}."))
            else:
                st.info(t("Kedua produk memiliki metrik tingkat risiko yang identik secara numerik.", "Both products have numerically identical risk level metrics."))

            fig_comp = go.Figure(data=[
                go.Bar(name=prod_a_display, x=[t('Skor Risiko AI (%)', 'AI Risk Score (%)')], y=[score_a], marker_color='#3498DB', text=[f"{score_a:.1f}%"], textposition='auto'),
                go.Bar(name=prod_b_display, x=[t('Skor Risiko AI (%)', 'AI Risk Score (%)')], y=[score_b], marker_color='#E74C3C', text=[f"{score_b:.1f}%"], textposition='auto')
            ])
            fig_comp.update_layout(barmode='group', title=t("Perbandingan Head-to-Head Skor Risiko", "Head-to-Head Risk Score Comparison"), height=400)
            st.plotly_chart(fig_comp, use_container_width=True)

            st.markdown("---")

        row1_colA, row1_colB = st.columns(2, gap="large")
        with row1_colA:
            st.markdown(f"### {t('Hasil:', 'Result:')} {res_a['product_name'] or t('Produk A', 'Product A')}")
            st.markdown(t("#### Profil & Analisis Produk Dasar", "#### Basic Product Profile & Analysis"))
            st.caption(t("Ringkasan prediksi risiko dan kontribusi nutrisi utama.", "Summary of predicted risk and main nutritional contributions."))
            render_analysis_side(res_a)
        with row1_colB:
            st.markdown(f"### {t('Hasil:', 'Result:')} {res_b['product_name'] or t('Produk B', 'Product B')}")
            st.markdown(t("#### Profil & Analisis Produk Dasar", "#### Basic Product Profile & Analysis"))
            st.caption(t("Ringkasan prediksi risiko dan kontribusi nutrisi utama.", "Summary of predicted risk and main nutritional contributions."))
            render_analysis_side(res_b)

        if res_a.get("status") == "ok" and res_b.get("status") == "ok":
            st.markdown("---")
            
            row2_colA, row2_colB = st.columns(2, gap="large")
            with row2_colA:
                risk_info_a = res_a.get("risk_info", classify_risk(res_a.get("risk_score", 0)))
                render_recommendation_details(risk_info_a, res_a.get("recommendation", ""), res_a.get("is_upf"), res_a.get("upf_flags", []))
            with row2_colB:
                risk_info_b = res_b.get("risk_info", classify_risk(res_b.get("risk_score", 0)))
                render_recommendation_details(risk_info_b, res_b.get("recommendation", ""), res_b.get("is_upf"), res_b.get("upf_flags", []))
                
            st.markdown("---")
            
            st.markdown(t("### 📊 Profil Gizi & Makronutrien Holistik", "### 📊 Holistic Nutrition & Macronutrient Profile"))
            st.caption(t("Analisis mendalam mengenai sumber kalori dan dampak glikemik berdasarkan takaran saji.", "In-depth analysis of calorie sources and glycemic impact based on serving size."))
            
            row3a_colA, row3a_colB = st.columns(2, gap="large")
            with row3a_colA:
                render_nutrition_kepadatan_gula(res_a["nutrition_data"], res_a["takaran_saji"])
            with row3a_colB:
                render_nutrition_kepadatan_gula(res_b["nutrition_data"], res_b["takaran_saji"])

            st.write("")
            st.markdown(t("**Distribusi Sumber Kalori (Macronutrient Split)**", "**Calorie Source Distribution (Macronutrient Split)**"))
            
            row3b_colA, row3b_colB = st.columns(2, gap="large")
            with row3b_colA:
                render_nutrition_pie_chart(res_a["nutrition_data"])
            with row3b_colB:
                render_nutrition_pie_chart(res_b["nutrition_data"])
                
            st.markdown("---")
            
            row4_colA, row4_colB = st.columns(2, gap="large")
            with row4_colA:
                render_health_metrics(res_a["nutrition_data"], res_a["takaran_saji"], current_threshold, show_header=True)
            with row4_colB:
                render_health_metrics(res_b["nutrition_data"], res_b["takaran_saji"], current_threshold, show_header=True)


elif app_mode in ["Simulasi Konsumsi Produk", "Consumption Simulation"]:
    st.header(t("Simulasi Konsumsi Produk", "Product Consumption Simulation"))
    st.info(t(
        "Masukkan detail produk dan perkirakan dampak risikonya berdasarkan frekuensi konsumsi Anda.",
        "Input product details and estimate its risk impact based on your consumption frequency."
    ))

    if "sim_data" not in st.session_state:
        st.session_state.sim_data = init_parsed_data()
    if "sim_ver" not in st.session_state:
        st.session_state.sim_ver = 0

    def apply_sim_preset():
        st.session_state.sim_data = dict(EXAMPLE_PRESETS[st.session_state.preset_sim])
        st.session_state.sim_ver += 1

    st.markdown(t("### Langkah 1: Definisikan Produk", "### Step 1: Define Product"))
    method_options = [t("Pilih Contoh Produk", "Choose Example"), t("Scan Label (OCR)", "Scan Label (OCR)"), t("Input Manual", "Manual Input")]
    method_sim = st.radio(t("Metode Input Produk:", "Product Input Method:"), method_options, horizontal=True, key="method_sim")
    
    if method_sim in ["Pilih Contoh Produk", "Choose Example"]:
        st.selectbox(t("Contoh Uji Produk", "Example Test Product"), list(EXAMPLE_PRESETS.keys()), key="preset_sim", format_func=lambda x: t(x, EN_PRESET_NAMES.get(x, x)), on_change=apply_sim_preset)
    elif method_sim in ["Scan Label (OCR)", "Scan Label (OCR)"]:
        col_scan_gizi_sim, col_scan_komp_sim = st.columns(2)
        with col_scan_gizi_sim:
            st.markdown(t("**1. Scan Nilai Gizi**", "**1. Scan Nutrition**"))
            type_sim_gizi = st.radio("S. Gizi Sim", [t("Upload", "Upload"), t("Kamera", "Camera")], key="type_sim_gizi", horizontal=True, label_visibility="collapsed")
            file_sim_gizi = st.file_uploader(t("Foto Gizi Sim", "Nutrition Photo Sim"), type=["jpg", "jpeg", "png"], key="file_sim_gizi") if type_sim_gizi in ["Upload"] else st.camera_input(t("Kamera Gizi Sim", "Nutrition Camera Sim"), key="cam_sim_gizi")
            if file_sim_gizi:
                img_sim_gizi = Image.open(file_sim_gizi)
                safe_image(standardize_image_size(img_sim_gizi), caption=t("Gizi Sim", "Nutrition Sim"), width=250)
                if st.button("🔍 OCR Gizi", key="btn_ocr_sim_gizi"):
                    with st.spinner(t("Membaca...", "Reading...")):
                        reader, err = get_ocr_reader_safely()
                        if err: st.error(err)
                        else:
                            res, err2 = run_ocr_safely(reader, img_sim_gizi, "nutrition")
                            if err2: st.error(err2)
                            else:
                                for k, v in res["parsed"].items():
                                    if v not in [0, 0.0, "Tidak terdeteksi.", "", "Produk Tanpa Nama"]:
                                        st.session_state.sim_data[k] = v
                        st.session_state.sim_ver += 1
                        st.success(t("Tersimpan!", "Saved!"))

        with col_scan_komp_sim:
            st.markdown(t("**2. Scan Komposisi**", "**2. Scan Composition**"))
            type_sim_komp = st.radio("S. Komp Sim", [t("Upload", "Upload"), t("Kamera", "Camera")], key="type_sim_komp", horizontal=True, label_visibility="collapsed")
            file_sim_komp = st.file_uploader(t("Foto Komp Sim", "Composition Photo Sim"), type=["jpg", "jpeg", "png"], key="file_sim_komp") if type_sim_komp in ["Upload"] else st.camera_input(t("Kamera Komp Sim", "Composition Camera Sim"), key="cam_sim_komp")
            if file_sim_komp:
                img_sim_komp = Image.open(file_sim_komp)
                safe_image(standardize_image_size(img_sim_komp), caption=t("Komp Sim", "Comp Sim"), width=250)
                if st.button("🔍 OCR Komposisi", key="btn_ocr_sim_komp"):
                    with st.spinner(t("Membaca...", "Reading...")):
                        reader, err = get_ocr_reader_safely()
                        if err: st.error(err)
                        else:
                            res, err2 = run_ocr_safely(reader, img_sim_komp, "composition")
                            if err2: st.error(err2)
                            else:
                                val = res["parsed"].get("komposisi", "")
                                if val and val != "Tidak terdeteksi.":
                                    st.session_state.sim_data["komposisi"] = val
                        st.session_state.sim_ver += 1
                        st.success(t("Tersimpan!", "Saved!"))

    prod_name_sim, saji_sim, nut_sim, kompo_sim = input_form(f"sim_{st.session_state.sim_ver}", st.session_state.sim_data)

    st.markdown("---")
    st.markdown(t("### Langkah 2: Atur Pola Konsumsi", "### Step 2: Set Consumption Pattern"))
    col_pola1, col_pola2 = st.columns(2)
    freq_weekly = col_pola1.slider(t("Frekuensi konsumsi per minggu (kali/sajian)", "Consumption frequency per week (times/servings)"), 1, 21, 3)
    sim_period = col_pola2.slider(t("Periode Simulasi (Bulan)", "Simulation Period (Months)"), 1, 12, 1)

    if st.button(t("🚀 Jalankan Simulasi", "🚀 Run Simulation"), type="primary", use_container_width=True):
        res_sim = build_analysis_result(prod_name_sim, saji_sim, nut_sim, kompo_sim)
        
        if res_sim.get("status") == "ok":
            st.markdown("---")
            st.markdown(t("### 📈 Hasil Simulasi", "### 📈 Simulation Results"))
            
            total_days = sim_period * 30
            total_weeks = total_days / 7
            total_servings = freq_weekly * total_weeks
            
            acc_sugar = float(nut_sim.get("gula", 0)) * total_servings
            acc_sodium = float(nut_sim.get("natrium", 0)) * total_servings
            acc_sat_fat = float(nut_sim.get("lemak_jenuh", 0)) * total_servings
            acc_cal = float(nut_sim.get("energi", 0)) * total_servings
            
            max_sugar = current_threshold["gula"] * total_days
            max_sodium = current_threshold["natrium"] * total_days
            max_sat_fat = current_threshold["lemak_jenuh"] * total_days
            max_cal = current_threshold["kalori"] * total_days

            st.markdown(t(f"#### Dampak Akumulatif Selama {sim_period} Bulan (~{total_days} hari)", f"#### Cumulative Impact Over {sim_period} Month(s) (~{total_days} days)"))
            c1, c2, c3 = st.columns(3)
            c1.metric(t("Total Sajian Dikonsumsi", "Total Servings Consumed"), f"{int(total_servings)} {t('porsi','servings')}", f"{freq_weekly}x {t('seminggu','a week')}")
            c2.metric(t("Total Gula dari Produk", "Total Sugar from Product"), f"{acc_sugar:.1f} g", t(f"Setara ~ {acc_sugar/15:.1f} sdm gula", f"Equals ~ {acc_sugar/15:.1f} tbsp sugar"))
            c3.metric(t("Total Kalori dari Produk", "Total Calories from Product"), f"{acc_cal:.1f} kkal", t(f"Setara ~ {acc_cal/7700:.1f} kg lemak", f"Equals ~ {acc_cal/7700:.1f} kg fat"))

            st.write("")
            st.markdown(t("##### ⚠️ Persentase Konsumsi Terhadap Batas Maksimal (Angka Kecukupan Gizi) Anda dalam Periode Ini:", "##### ⚠️ Consumption Percentage Against Your Maximum Limit (RDA) in This Period:"))
            
            gula_pct = (acc_sugar / max_sugar * 100) if max_sugar else 0
            natrium_pct = (acc_sodium / max_sodium * 100) if max_sodium else 0
            lemak_jenuh_pct = (acc_sat_fat / max_sat_fat * 100) if max_sat_fat else 0
            kalori_pct = (acc_cal / max_cal * 100) if max_cal else 0

            custom_progress_bar(t("Kalori yang Dihabiskan", "Calories Consumed"), acc_cal, max_cal, "kkal", "#10B981", kalori_pct)
            custom_progress_bar(t("Batas Gula yang Dihabiskan", "Sugar Limit Consumed"), acc_sugar, max_sugar, "g", "#F59E0B", gula_pct)
            custom_progress_bar(t("Batas Natrium yang Dihabiskan", "Sodium Limit Consumed"), acc_sodium, max_sodium, "mg", "#3498DB", natrium_pct)
            custom_progress_bar(t("Batas Lemak Jenuh yang Dihabiskan", "Saturated Fat Limit Consumed"), acc_sat_fat, max_sat_fat, "g", "#9B59B6", lemak_jenuh_pct)
            
            st.info(t(
                "Simulasi di atas menunjukkan seberapa besar jatah nutrisi Anda yang **habis hanya oleh satu jenis produk ini saja** selama periode simulasi. Idealnya, camilan atau minuman tunggal tidak boleh mendominasi batas asupan harian/bulanan Anda.",
                "The simulation above shows how much of your nutrient allowance is **consumed solely by this one product type** over the simulation period. Ideally, a single snack or beverage should not dominate your daily/monthly limits."
            ))
            
            st.markdown("---")
            col_simA, col_simB = st.columns(2, gap="large")
            with col_simA:
                st.markdown(t("### 📋 Profil & Analisis Produk Dasar", "### 📋 Basic Product Profile & Analysis"))
                st.caption(t("Ringkasan prediksi risiko dan kontribusi nutrisi utama.", "Summary of predicted risk and main nutritional contributions."))
                render_analysis_side(res_sim)
            with col_simB:
                render_holistic_nutrition_profile(res_sim["nutrition_data"], res_sim["takaran_saji"])
                
            st.markdown("---")
            risk_info_sim = res_sim.get("risk_info", classify_risk(res_sim.get("risk_score", 0)))
            render_recommendation_details(risk_info_sim, res_sim.get("recommendation", ""), res_sim.get("is_upf"), res_sim.get("upf_flags", []))
            
            st.markdown("---")
            render_health_metrics(res_sim["nutrition_data"], res_sim["takaran_saji"], current_threshold, show_header=True)
        else:
            st.error(t("Silakan lengkapi data nutrisi produk untuk menjalankan simulasi.", "Please complete product nutrition data to run simulation."))


elif app_mode in ["Riwayat Analisis", "Analysis History"]:
    st.header(t("Riwayat Analisis", "Analysis History"))
    st.write(t("Daftar lengkap riwayat analisis produk yang dilakukan pada sesi ini.", "Complete list of product analysis history performed in this session."))
    
    if st.button(t("🗑️ Hapus Riwayat", "🗑️ Clear History")):
        st.session_state.scan_history = []
        st.success(t("Riwayat berhasil dihapus!", "History successfully cleared!"))

    if st.session_state.scan_history:
        valid_history = []
        for item in st.session_state.scan_history:
            if item.get('product_name') is None and item.get('Nama Produk') is None:
                continue
            
            if 'date' in item:
                nut_data = item.get('nutrition', {})
                if isinstance(nut_data, str): 
                    nut_data = {}
                valid_history.append({
                    "Waktu Analisis": item.get("date"),
                    "Nama Produk": item.get("product_name"),
                    "Skor Risiko (%)": item.get("risk_score"),
                    "Klasifikasi": item.get("classification"),
                    "Takaran Saji (g/ml)": 100, 
                    "Energi (kkal)": nut_data.get("energi", 0) if isinstance(nut_data, dict) else 0,
                    "Lemak Total (g)": nut_data.get("lemak_total", 0) if isinstance(nut_data, dict) else 0,
                    "Lemak Jenuh (g)": nut_data.get("lemak_jenuh", 0) if isinstance(nut_data, dict) else 0,
                    "Protein (g)": nut_data.get("protein", 0) if isinstance(nut_data, dict) else 0,
                    "Karbohidrat (g)": nut_data.get("karbohidrat", 0) if isinstance(nut_data, dict) else 0,
                    "Gula (g)": nut_data.get("gula", 0) if isinstance(nut_data, dict) else 0,
                    "Garam (g)": nut_data.get("garam", 0) if isinstance(nut_data, dict) else 0,
                    "Natrium (mg)": nut_data.get("natrium", 0) if isinstance(nut_data, dict) else 0,
                    "Natrium Benzoat (mg)": nut_data.get("natrium_benzoat", 0) if isinstance(nut_data, dict) else 0,
                    "Komposisi": "-",
                    "Rekomendasi": "-"
                })
            else:
                valid_history.append(item)
        
        st.session_state.scan_history = valid_history
        
        if valid_history:
            df_hist = pd.DataFrame(valid_history)

            # Display translated dataframe based on language
            df_disp_hist = df_hist.copy()
            if st.session_state.lang == "EN":
                df_disp_hist.rename(columns={
                    "Waktu Analisis": "Analysis Time",
                    "Nama Produk": "Product Name",
                    "Skor Risiko (%)": "Risk Score (%)",
                    "Klasifikasi": "Classification",
                    "Takaran Saji (g/ml)": "Serving Size (g/ml)",
                    "Energi (kkal)": "Energy (kcal)",
                    "Lemak Total (g)": "Total Fat (g)",
                    "Lemak Jenuh (g)": "Saturated Fat (g)",
                    "Protein (g)": "Protein (g)",
                    "Karbohidrat (g)": "Carbohydrate (g)",
                    "Gula (g)": "Sugar (g)",
                    "Garam (g)": "Salt (g)",
                    "Natrium (mg)": "Sodium (mg)",
                    "Natrium Benzoat (mg)": "Sodium Benzoate (mg)",
                    "Komposisi": "Composition",
                    "Rekomendasi": "Recommendation"
                }, inplace=True)
                df_disp_hist["Classification"] = df_disp_hist["Classification"].apply(tr_risk)

            st.dataframe(df_disp_hist, use_container_width=True)
            
            st.markdown("---")
            output_hist = io.BytesIO()
            with pd.ExcelWriter(output_hist, engine="openpyxl") as writer:
                df_disp_hist.to_excel(writer, index=False, sheet_name=t("Riwayat Analisis", "Analysis History"))
            
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label=t("📊 Download Riwayat (Excel)", "📊 Download History (Excel)"), 
                    data=output_hist.getvalue(), 
                    file_name="riwayat_analisis_nutriscan.xlsx",
                    use_container_width=True
                )
            
            with col_dl2:
                try:
                    from fpdf import FPDF
                    pdf_ready = True
                except ImportError:
                    pdf_ready = False
                
                if pdf_ready:
                    with st.spinner(t("Menyiapkan dokumen PDF...", "Preparing PDF document...")):
                        pdf_hist_data = generate_history_pdf_report(df_hist)
                        st.download_button(
                            label=t("📑 Download Riwayat (PDF)", "📑 Download History (PDF)"), 
                            data=pdf_hist_data, 
                            file_name="Riwayat_Analisis_NutriScan.pdf", 
                            mime="application/pdf",
                            use_container_width=True
                        )
                else:
                    st.warning(t("⚠️ Modul 'fpdf' belum terinstall. Jalankan `pip install fpdf` di terminal.", "⚠️ Module 'fpdf' not installed. Run `pip install fpdf`."))
        else:
            st.info(t("Belum ada riwayat analisis pada sesi ini.", "No analysis history in this session yet."))
    else:
        st.info(t("Belum ada riwayat analisis pada sesi ini.", "No analysis history in this session yet."))


elif app_mode in ["Edukasi Gizi", "Nutrition Education"]:
    st.header(t("Edukasi Gizi", "Nutrition Education"))
    st.markdown(t(
        """
        **Cara membaca hasil aplikasi:**

        1. OCR hanya membantu mengisi data awal, bukan pengganti validasi pengguna.
        2. Data kosong tidak dianalisis agar aplikasi tidak memberi klasifikasi palsu.
        3. Klasifikasi Aman, Sedang, dan Tinggi memakai satu fungsi keputusan.
        4. Gula tinggi perlu diperhatikan karena berpengaruh pada beban asupan harian.
        5. Natrium tinggi perlu dibatasi, terutama pada pengguna dengan risiko hipertensi.
        6. Lemak jenuh tinggi sebaiknya tidak dikonsumsi terlalu sering.
        7. Komposisi dengan pemanis buatan, pewarna sintetik, pengawet, dan penguat rasa menandakan indikasi produk ultra proses.
        """,
        """
        **How to read the application results:**

        1. OCR only helps fill in initial data, it is not a substitute for user validation.
        2. Empty data is not analyzed so the application doesn't provide false classifications.
        3. Safe, Moderate, and High classifications use a single decision function.
        4. High sugar must be monitored as it affects the daily intake load.
        5. High sodium needs to be limited, especially for users at risk for hypertension.
        6. High saturated fat should not be consumed too frequently.
        7. Composition with artificial sweeteners, synthetic colors, preservatives, and flavor enhancers indicates an ultra-processed product.
        """
    ))