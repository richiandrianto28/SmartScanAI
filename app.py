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
        with st.expander(f"Catatan kualitas OCR {label}", expanded=False):
            for item in quality_warnings:
                st.warning(item)

    if errors:
        with st.expander(f"Catatan error OCR {label}", expanded=False):
            for item in errors:
                st.warning(item)

    with st.expander(f"Lihat teks OCR {label}", expanded=False):
        st.caption(f"Variasi gambar terbaik: {scan_result.get('best_variant', 'tidak diketahui')}")
        st.text(scan_result.get("raw_text") or "Tidak ada teks terbaca")

    with st.expander(f"Ringkasan variasi preprocessing {label}", expanded=False):
        names = list(scan_result.get("variants", {}).keys())
        st.write(names if names else "Tidak ada variasi gambar tersimpan.")


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
    if gender == "Pria":
        bmr = (10 * berat) + (6.25 * tinggi) - (5 * usia) + 5
    else:
        bmr = (10 * berat) + (6.25 * tinggi) - (5 * usia) - 161

    faktor = {
        "Sedentary": 1.2,
        "Ringan": 1.375,
        "Sedang": 1.55,
        "Aktif": 1.725,
        "Sangat Aktif": 1.9,
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
    st.metric("Skor Risiko Prediksi", f"{risk_score:.2f}%")
    st.metric("Klasifikasi", risk_info["label"])

    if risk_info["style"] == "success":
        st.success(f"Klasifikasi {risk_info['label']}")
    elif risk_info["style"] == "warning":
        st.warning(f"Klasifikasi {risk_info['label']}")
    else:
        st.error(f"Klasifikasi {risk_info['label']}")


def render_xai_radar(xai_factors):
    categories = list(xai_factors.keys())
    if not categories:
        return

    norm_values = []
    for key, value in xai_factors.items():
        key_lower = key.lower()
        value = float(value or 0)
        if "gula" in key_lower:
            norm_values.append(min((value / 45) * 100, 100))
        elif "natrium" in key_lower and "benzoat" not in key_lower:
            norm_values.append(min((value / 1500) * 100, 100))
        elif "benzoat" in key_lower:
            norm_values.append(min((value / 300) * 100, 100))
        elif "lemak total" in key_lower:
            norm_values.append(min((value / 30) * 100, 100))
        elif "lemak jenuh" in key_lower:
            norm_values.append(min((value / 14) * 100, 100))
        elif "energi" in key_lower:
            norm_values.append(min((value / 550) * 100, 100))
        elif "karbohidrat" in key_lower:
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
            name="Kandungan Produk",
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
    st.markdown("### Rekomendasi")
    
    if risk_info["style"] == "success":
        st.info(f"{recommendation_text}")
    elif risk_info["style"] == "warning":
        st.warning(f"{recommendation_text}")
    else:
        st.error(f"{recommendation_text}")

    if is_upf:
        st.error("Indikasi bahan ultra proses terdeteksi")
        st.write(", ".join(upf_flags))
        
    st.markdown("---")
    
    with st.expander("ℹ️ Detail Penjelasan Klasifikasi Nutrisi", expanded=False):
        st.markdown("""
        * 🟢 **Aman (0 - 34.99):** Produk relatif aman dan sehat. Cocok untuk dikonsumsi dalam porsi wajar sebagai bagian dari asupan nutrisi harian Anda.
        * 🟡 **Sedang (35 - 69.99):** Kandungan produk memiliki beberapa catatan (misal: kalori cukup padat atau ada gula tambahan). Boleh dikonsumsi sesekali, namun bukan untuk konsumsi utama harian yang berulang-ulang.
        * 🔴 **Tinggi (70 - 100):** Sangat disarankan untuk dibatasi. Produk ini kemungkinan besar padat energi tanpa nutrisi bermanfaat (empty calories), tinggi gula/garam, atau merupakan produk *ultra-processed*.
        """)


def render_nutrition_kepadatan_gula(nutrition_data, takaran_saji):
    energi = float(nutrition_data.get("energi", 0))
    gula = float(nutrition_data.get("gula", 0))
    karbohidrat = float(nutrition_data.get("karbohidrat", 0))

    kepadatan = energi / takaran_saji if takaran_saji > 0 else 0
    rasio_gula = (gula / karbohidrat * 100) if karbohidrat > 0 else 0

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Kepadatan Energi (kkal/gram)**")
        st.markdown(f"## {kepadatan:.1f}")
        if kepadatan > 4.0:
            st.error("↑ 🔴 Sangat Tinggi (Padat Kalori)")
            st.caption("Menunjukkan seberapa padat kalori dalam produk ini. Kepadatan tinggi memicu obesitas jika tidak dikontrol.")
        elif kepadatan >= 1.5:
            st.warning("— 🟡 Sedang")
            st.caption("Kepadatan kalori moderat. Perhatikan porsi konsumsi Anda.")
        else:
            st.success("↓ 🟢 Rendah Kalori")
            st.caption("Produk ini memiliki kepadatan energi yang rendah, baik untuk mengontrol asupan kalori.")

    with col2:
        st.markdown("**Rasio Gula dari Total Karbohidrat**")
        st.markdown(f"## {rasio_gula:.1f}%")
        if rasio_gula > 50:
            st.error("↑ 🔴 Tinggi Gula Sederhana")
            st.caption("Jika >50%, sebagian besar karbohidrat adalah gula sederhana yang bisa memicu lonjakan gula darah (*sugar spike*).")
        elif rasio_gula >= 20:
            st.warning("— 🟡 Sedang")
            st.caption("Mengandung gula sederhana dalam jumlah sedang.")
        else:
            st.success("↓ 🟢 Rendah Gula")
            st.caption("Sebagian besar karbohidrat berasal dari sumber kompleks yang lebih lama dicerna.")


def render_nutrition_pie_chart(nutrition_data):
    lemak_total = float(nutrition_data.get("lemak_total", 0))
    karbohidrat = float(nutrition_data.get("karbohidrat", 0))
    protein = float(nutrition_data.get("protein", 0))

    kalori_lemak = lemak_total * 9
    kalori_karbo = karbohidrat * 4
    kalori_protein = protein * 4
    total_kal_makro = kalori_lemak + kalori_karbo + kalori_protein

    if total_kal_makro > 0:
        labels = ['Lemak (9 kkal/g)', 'Karbohidrat (4 kkal/g)', 'Protein (4 kkal/g)']
        values = [kalori_lemak, kalori_karbo, kalori_protein]
        colors = ['#E74C3C', '#2ECC71', '#3498DB'] 

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, marker=dict(colors=colors))])
        fig.update_traces(textinfo='percent+label', textposition='inside')
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Data makronutrien kosong atau bernilai nol. Isi Lemak, Karbohidrat, dan Protein untuk melihat rasio kalori.")


def render_holistic_nutrition_profile(nutrition_data, takaran_saji):
    st.markdown("### 📊 Profil Gizi & Makronutrien Holistik")
    st.caption("Analisis mendalam mengenai sumber kalori dan dampak glikemik berdasarkan takaran saji.")
    render_nutrition_kepadatan_gula(nutrition_data, takaran_saji)
    st.write("")
    st.markdown("**Distribusi Sumber Kalori (Macronutrient Split)**")
    render_nutrition_pie_chart(nutrition_data)


def custom_progress_bar(label, current_val, max_val, unit, color, percentage):
    display_pct = min(percentage, 100)
    
    warning_text = ""
    if percentage > 100:
        color = "#E74C3C" 
        warning_text = "<span style='color:#E74C3C; font-weight:bold; font-size: 0.9em; margin-left: 5px;'>(Melebihi Batas!)</span>"

    html_code = f"<div style='margin-bottom: 24px; font-family: \"Inter\", \"Segoe UI\", sans-serif;'><div style='display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 8px;'><span style='font-weight: 600; font-size: 15px; color: #0F172A;'>{label}</span><span style='color: #475569; font-size: 14px;'><span style='font-weight: 700; color: #1E293B;'>{current_val:.2f}</span> / {max_val:.2f} {unit} <span style='color: #64748B; margin-left: 4px;'>({percentage:.1f}%)</span>{warning_text}</span></div><div style='width: 100%; background-color: #E2E8F0; border-radius: 8px; height: 14px; box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);'><div style='width: {display_pct}%; background-color: {color}; height: 100%; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.1); transition: width 0.8s ease-out;'></div></div></div>"
    st.markdown(html_code, unsafe_allow_html=True)


def render_health_metrics(nutrition_data, takaran_saji, current_threshold, show_header=True):
    if show_header:
        st.markdown("### 🎯 Pemenuhan Angka Kecukupan Gizi Harian")
        st.caption("Berdasarkan profil pengguna dan batas ambang kesehatan medis Anda:")
        st.write("")

    gula = float(nutrition_data.get("gula", 0))
    natrium = float(nutrition_data.get("natrium", 0))
    lemak_jenuh = float(nutrition_data.get("lemak_jenuh", 0))

    gula_pct = (gula / current_threshold["gula"] * 100) if current_threshold["gula"] else 0
    natrium_pct = (natrium / current_threshold["natrium"] * 100) if current_threshold["natrium"] else 0
    lemak_jenuh_pct = (lemak_jenuh / current_threshold["lemak_jenuh"] * 100) if current_threshold["lemak_jenuh"] else 0

    custom_progress_bar("Gula", gula, current_threshold["gula"], "g", "#F59E0B", gula_pct)
    custom_progress_bar("Natrium", natrium, current_threshold["natrium"], "mg", "#3498DB", natrium_pct)
    custom_progress_bar("Lemak Jenuh", lemak_jenuh, current_threshold["lemak_jenuh"], "g", "#9B59B6", lemak_jenuh_pct)


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
            "message": "Data belum cukup untuk dianalisis. Isi atau koreksi minimal satu nilai gizi yang valid sebelum menjalankan rekomendasi.",
            "integrity_note": "Sistem tidak memberi label tinggi, sedang, atau aman ketika data masih kosong. Ini menjaga integritas hasil analisis.",
            "product_name": product_name or "Produk Tanpa Nama",
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
        "product_name": product_name or "Produk Tanpa Nama",
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


# FITUR AI Insight Generator
def generate_batch_insights(df_results):
    if df_results is None or df_results.empty:
        return "Belum ada data untuk dianalisis."

    valid_df = df_results.copy()
    valid_df["Skor Risiko Numerik"] = pd.to_numeric(valid_df["Skor Risiko"], errors='coerce')
    valid_df = valid_df.dropna(subset=["Skor Risiko Numerik"])

    if valid_df.empty:
        return "Data tidak valid untuk membuat ringkasan AI."

    total_products = len(valid_df)
    aman_count = len(valid_df[valid_df["Klasifikasi"] == "Aman"])
    pct_aman = (aman_count / total_products) * 100 if total_products > 0 else 0

    highest_risk_row = valid_df.loc[valid_df["Skor Risiko Numerik"].idxmax()]
    highest_name = highest_risk_row.get("Nama Produk", "Produk Tidak Diketahui")
    highest_score = highest_risk_row["Skor Risiko Numerik"]

    avg_score = valid_df["Skor Risiko Numerik"].mean()
    if avg_score < 35:
        avg_cat = "rendah hingga sedang"
    elif avg_score < 70:
        avg_cat = "sedang hingga tinggi"
    else:
        avg_cat = "tinggi"

    reason_text = "kandungan komposisi gizinya"
    if "nutrition_data" in highest_risk_row and isinstance(highest_risk_row["nutrition_data"], dict):
        nut_data = highest_risk_row["nutrition_data"]
        high_factors = []
        
        if float(nut_data.get("natrium", 0)) > 300: high_factors.append("natrium")
        if float(nut_data.get("gula", 0)) > 12: high_factors.append("gula")
        if float(nut_data.get("lemak_total", 0)) > 10: high_factors.append("lemak")
        if float(nut_data.get("lemak_jenuh", 0)) > 4: high_factors.append("lemak jenuh")
        
        if high_factors:
            if len(high_factors) > 1:
                reason_text = f"kandungan {', '.join(high_factors[:-1])} dan {high_factors[-1]} yang relatif tinggi"
            else:
                reason_text = f"kandungan {high_factors[0]} yang relatif tinggi"

    insight = (
        f"Dari **{total_products} produk** yang dianalisis, **{pct_aman:.1f}%** termasuk kategori aman. "
        f"Produk dengan skor risiko tertinggi adalah **{highest_name}** ({highest_score:.2f}), "
        f"terutama dipengaruhi oleh {reason_text}. "
        f"Secara keseluruhan, rata-rata skor risiko batch ini adalah **{avg_score:.1f}** yang menunjukkan "
        f"tingkat risiko konsumsi berada pada kategori **{avg_cat}**."
    )
    return insight


# FITUR Export Report PDF
def generate_pdf_report(df_results, insight_text, fig_pie):
    from fpdf import FPDF
    
    class PDFReport(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 16)
            self.set_text_color(15, 23, 42)
            self.cell(0, 10, 'Laporan Analisis Batch - SMART NutriScan AI', 0, 1, 'C')
            self.set_draw_color(200, 200, 200)
            self.line(10, 22, 200, 22)
            self.ln(10)
            
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, f'Halaman {self.page_no()}', 0, 0, 'C')

    pdf = PDFReport()
    pdf.add_page()
    
    # 1. Ringkasan Eksekutif
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 10, "1. Ringkasan Eksekutif", 0, 1)
    
    pdf.set_font("Arial", '', 11)
    pdf.set_text_color(50, 50, 50)
    clean_insight = insight_text.replace('**', '').replace('\n', ' ')
    clean_insight = clean_insight.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 7, clean_insight)
    pdf.ln(5)
    
    # 2. Statistik
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 10, "2. Statistik Distribusi Risiko", 0, 1)
    
    valid_df = df_results.dropna(subset=["Skor Risiko Numerik"])
    total = len(valid_df)
    
    if total > 0:
        aman = len(valid_df[valid_df["Klasifikasi"] == "Aman"])
        sedang = len(valid_df[valid_df["Klasifikasi"] == "Sedang"])
        tinggi = len(valid_df[valid_df["Klasifikasi"] == "Tinggi"])
        avg_score = valid_df["Skor Risiko Numerik"].mean()
        
        pdf.set_font("Arial", '', 11)
        pdf.set_text_color(50, 50, 50)
        stats_text = (
            f"- Total Produk Dianalisis: {total} produk\n"
            f"- Kategori Aman: {aman} produk ({(aman/total)*100:.1f}%)\n"
            f"- Kategori Sedang: {sedang} produk ({(sedang/total)*100:.1f}%)\n"
            f"- Kategori Tinggi: {tinggi} produk ({(tinggi/total)*100:.1f}%)\n"
            f"- Rata-rata Skor Risiko Keseluruhan: {avg_score:.2f} / 100"
        )
        pdf.multi_cell(0, 7, stats_text)
    pdf.ln(5)
    
    # 3. Grafik Proporsi Klasifikasi
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 10, "3. Grafik Proporsi Klasifikasi", 0, 1)
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            fig_pie.write_image(tmpfile.name, format="png", width=600, height=400)
            pdf.image(tmpfile.name, x=45, w=120)
        os.unlink(tmpfile.name)
        pdf.ln(5)
    except Exception as e:
        pdf.set_font("Arial", 'I', 10)
        pdf.set_text_color(200, 50, 50)
        pdf.cell(0, 10, "(Grafik tidak dapat diekspor. Pastikan package 'kaleido' terinstall di backend.)", 0, 1)
        pdf.ln(5)
        
    # 4. Tabel Hasil Analisis
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 10, "4. Detail Tabel Hasil Analisis", 0, 1)
    pdf.ln(2)
    
    # Table Header
    pdf.set_font("Arial", 'B', 10)
    pdf.set_fill_color(240, 244, 248)
    col_w = [110, 35, 45]
    pdf.cell(col_w[0], 10, "Nama Produk", 1, 0, 'C', fill=True)
    pdf.cell(col_w[1], 10, "Skor Risiko", 1, 0, 'C', fill=True)
    pdf.cell(col_w[2], 10, "Klasifikasi", 1, 1, 'C', fill=True)
    
    # Table Content
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(0, 0, 0)
    
    for _, row in valid_df.sort_values(by="Skor Risiko Numerik", ascending=False).iterrows():
        name = str(row['Nama Produk']).strip()
        if len(name) > 55: name = name[:52] + "..."
        skor = f"{row['Skor Risiko Numerik']:.2f}%"
        klas = str(row['Klasifikasi'])
        
        name = name.encode('latin-1', 'replace').decode('latin-1')
        
        pdf.cell(col_w[0], 10, f" {name}", 1, 0, 'L')
        pdf.cell(col_w[1], 10, skor, 1, 0, 'C')
        pdf.cell(col_w[2], 10, klas, 1, 1, 'C')

    # Convert PDF object to Bytes safely
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmpdf:
        pdf.output(tmpdf.name)
        with open(tmpdf.name, 'rb') as f:
            pdf_bytes = f.read()
    os.unlink(tmpdf.name)
    
    return pdf_bytes


def render_analysis_side(analysis_result, current_signature=None):
    if not analysis_result:
        st.info("Hasil analisis akan muncul di sini setelah tombol analisis diklik.")
        return

    stored_signature = analysis_result.get("input_signature")
    if stored_signature is not None and current_signature is not None and stored_signature != current_signature:
        st.warning("Data input sudah berubah setelah analisis terakhir. Klik tombol analisis lagi untuk memperbarui hasil.")

    if analysis_result.get("status") == "insufficient":
        st.warning(analysis_result.get("message", "Data belum cukup untuk dianalisis."))
        st.info(analysis_result.get("integrity_note", "Periksa kembali data input sebelum analisis."))
        return

    risk_score = float(analysis_result.get("risk_score", 0))
    xai_factors = analysis_result.get("xai_factors", {})

    render_risk_status(risk_score)
    st.markdown("#### Radar Kontribusi Nutrisi")
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
        st.session_state.scan_history.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "product_name": analysis_result["product_name"],
            "risk_score": round(risk_score, 2),
            "classification": risk_info["label"],
            "nutrition": nutrition_data,
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

    product_name = st.text_input("Nama Produk", key=name_key)

    c0, c1, c2 = st.columns(3)
    takaran_saji = c0.number_input("Takaran Saji g atau ml", min_value=1.0, format="%.2f", key=saji_key)
    energi = c1.number_input("Energi kkal", min_value=0.0, format="%.2f", key=energi_key)
    lemak_total = c2.number_input("Lemak Total g", min_value=0.0, format="%.2f", key=lemak_key)

    c3, c4, c5 = st.columns(3)
    lemak_jenuh = c3.number_input("Lemak Jenuh g", min_value=0.0, format="%.2f", key=jenuh_key)
    protein = c4.number_input("Protein g", min_value=0.0, format="%.2f", key=protein_key)
    karbohidrat = c5.number_input("Karbohidrat g", min_value=0.0, format="%.2f", key=karbo_key)

    c6, c7, c8, c9 = st.columns(4)
    gula = c6.number_input("Gula g", min_value=0.0, format="%.2f", key=gula_key)
    garam = c7.number_input("Garam g", min_value=0.0, format="%.2f", key=garam_key)
    natrium = c8.number_input("Natrium mg", min_value=0.0, format="%.2f", key=natrium_key)
    natrium_benzoat = c9.number_input("Natrium Benzoat mg", min_value=0.0, format="%.2f", key=benzoat_key)

    komposisi = st.text_area("Komposisi", height=120, key=komposisi_key)

    nutrition_data = build_nutrition_data(
        energi, lemak_total, lemak_jenuh, protein, karbohidrat, gula, garam, natrium, natrium_benzoat
    )

    return product_name, takaran_saji, nutrition_data, komposisi


with st.sidebar:
    try:
        st.image("assets/Logo Smart NutriScan AI.png", width=150)
    except Exception:
        st.markdown("## SMART NutriScan AI")

    st.title("SMART NutriScan AI")
    st.header("Profil Pengguna")

    col_gender, col_age = st.columns(2)
    user_gender = col_gender.selectbox("Gender", ["Pria", "Wanita"])
    user_age = col_age.number_input("Usia", min_value=1, max_value=120, value=25)

    col_weight, col_height = st.columns(2)
    user_weight = col_weight.number_input("Berat kg", min_value=10.0, max_value=300.0, value=65.0)
    user_height = col_height.number_input("Tinggi cm", min_value=50.0, max_value=250.0, value=165.0)

    user_activity = st.selectbox("Aktivitas", ["Sedentary", "Ringan", "Sedang", "Aktif", "Sangat Aktif"])
    kondisi_medis = st.selectbox("Kondisi Khusus", ["Tidak Ada", "Penderita Hipertensi", "Risiko Penyakit Ginjal", "Anak anak"])

    current_threshold = hitung_tdee_dinamis(user_gender, user_age, user_weight, user_height, user_activity)
    if kondisi_medis == "Penderita Hipertensi":
        current_threshold["natrium"] = 1200
    elif kondisi_medis == "Risiko Penyakit Ginjal":
        current_threshold["natrium"] = 1000
        current_threshold["kalori"] *= 0.9
    elif kondisi_medis == "Anak anak":
        current_threshold["gula"] = 25
        current_threshold["natrium"] = 1500

    with st.expander("Lihat batas harian"):
        st.write(f"Kalori: {current_threshold['kalori']:.2f} kkal")
        st.write(f"Gula: {current_threshold['gula']:.2f} g")
        st.write(f"Lemak jenuh: {current_threshold['lemak_jenuh']:.2f} g")
        st.write(f"Natrium: {current_threshold['natrium']:.2f} mg")

    app_mode = st.radio(
        "Pilih Fitur",
        [
            "Analisis Produk Tunggal",
            "Scan from Image",
            "Analisis Batch Excel",
            "Perbandingan Produk",  
            "Simulasi Konsumsi Produk",
            "Riwayat Analisis",
            "Edukasi Gizi",
        ],
    )


st.title("SMART NutriScan AI")
st.caption("Analisis produk pangan berbasis OCR, machine learning, aturan gizi terkalibrasi, dan konfirmasi data manual.")

model_ready = all([feat_model, lgbm_model, w2v_model, scaler])
if model_ready:
    st.success("Model utama berhasil dimuat. Skor tetap dijaga oleh aturan gizi agar klasifikasi konsisten.")
else:
    st.warning("Sebagian model utama belum terbaca. Aplikasi tetap berjalan dengan analisis gizi terkalibrasi.")


if app_mode == "Analisis Produk Tunggal":
    st.header("Analisis Produk Tunggal")

    manual_input_col, manual_result_col = st.columns([1.15, 1], gap="large")

    with manual_input_col:
        st.subheader("Input Informasi Produk")
        
        if "preset_selector" not in st.session_state:
            st.session_state.preset_selector = list(EXAMPLE_PRESETS.keys())[0]

        st.selectbox(
            "Pilih contoh uji atau isi manual", 
            list(EXAMPLE_PRESETS.keys()), 
            key="preset_selector",
            on_change=apply_manual_preset
        )
        
        product_name, takaran_saji, nutrition_data, komposisi = input_form("manual", EXAMPLE_PRESETS["Kosong"])
        manual_signature = make_analysis_signature(product_name, takaran_saji, nutrition_data, komposisi)

        if st.button("Analisis AI dan Gizi", type="primary"):
            store_product_analysis_result(
                product_name,
                takaran_saji,
                nutrition_data,
                komposisi,
                store_key="manual_analysis_result",
                input_signature=manual_signature,
            )
            st.success("Analisis berhasil diperbarui. Hasil ditampilkan di panel kanan.")

    with manual_result_col:
        st.markdown("### Profil & Analisis Produk Dasar")
        st.caption("Ringkasan prediksi risiko dan kontribusi nutrisi utama.")
        render_analysis_side(st.session_state.manual_analysis_result, current_signature=manual_signature)

    render_analysis_bottom(st.session_state.manual_analysis_result, current_threshold)


elif app_mode == "Scan from Image":
    st.header("Scan Produk Otomatis")
    st.info("Ambil foto dekat, lurus, tidak blur, dan pastikan label memenuhi sebagian besar area gambar. Setelah OCR selesai, koreksi data sebelum analisis.")

    if st.button("Reset Hasil OCR"):
        st.session_state.ocr_data = init_parsed_data()
        clear_ocr_analysis_result()
        bump_ocr_form_version()
        st.success("Hasil OCR dan analisis terakhir sudah dikosongkan.")

    col_scan1, col_scan2 = st.columns(2)

    with col_scan1:
        st.subheader("Scan 1: Informasi Nilai Gizi")
        input_type_1 = st.radio("Metode input nilai gizi", ["Upload File", "Kamera Langsung"], key="input_gizi")
        img_file_1 = st.file_uploader("Upload foto nilai gizi", type=["jpg", "jpeg", "png"], key="upload_gizi") if input_type_1 == "Upload File" else st.camera_input("Foto nilai gizi", key="camera_gizi")

        if img_file_1 is not None:
            try:
                image_1_original = Image.open(img_file_1)
                image_1_display = standardize_image_size(image_1_original, target_ratio=4/3)
                safe_image(image_1_display, caption="Gambar nilai gizi", width=350)

                if st.button("Proses OCR Nilai Gizi", key="btn_ocr_gizi"):
                    with st.spinner("Mempersiapkan OCR dan membaca nilai gizi secara bertahap..."):
                        reader, reader_error = get_ocr_reader_safely()
                        if reader_error:
                            scan_result_1, ocr_error_1 = None, reader_error
                        else:
                            scan_result_1, ocr_error_1 = run_ocr_safely(reader, image_1_original, mode="nutrition")

                    if ocr_error_1:
                        st.error("OCR nilai gizi gagal diproses. Aplikasi tidak dihentikan. Silakan input manual atau coba foto yang lebih jelas.")
                        with st.expander("Detail error OCR nilai gizi"):
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

                        st.success("Nilai gizi berhasil diproses. Angka satuan g yang terbaca sebagai 9 sudah dikoreksi sebelum masuk form. Periksa lagi sebelum analisis.")
                        render_ocr_result_debug(scan_result_1, "nilai gizi")
            except Exception as exc:
                st.error("Gambar nilai gizi tidak bisa dibaca. Coba upload ulang dalam format JPG atau PNG.")
                with st.expander("Detail error gambar nilai gizi"):
                    st.code(str(exc))

    with col_scan2:
        st.subheader("Scan 2: Komposisi Produk")
        input_type_2 = st.radio("Metode input komposisi", ["Upload File", "Kamera Langsung"], key="input_komposisi")
        img_file_2 = st.file_uploader("Upload foto komposisi", type=["jpg", "jpeg", "png"], key="upload_komposisi") if input_type_2 == "Upload File" else st.camera_input("Foto komposisi", key="camera_komposisi")

        if img_file_2 is not None:
            try:
                image_2_original = Image.open(img_file_2)
                image_2_display = standardize_image_size(image_2_original, target_ratio=4/3)
                safe_image(image_2_display, caption="Gambar komposisi", width=350)

                if st.button("Proses OCR Komposisi", key="btn_ocr_komposisi"):
                    with st.spinner("Mempersiapkan OCR dan membaca komposisi secara bertahap..."):
                        reader, reader_error = get_ocr_reader_safely()
                        if reader_error:
                            scan_result_2, ocr_error_2 = None, reader_error
                        else:
                            scan_result_2, ocr_error_2 = run_ocr_safely(reader, image_2_original, mode="composition")

                    if ocr_error_2:
                        st.error("OCR komposisi gagal diproses. Aplikasi tidak dihentikan. Silakan input manual atau coba foto yang lebih jelas.")
                        with st.expander("Detail error OCR komposisi"):
                            st.code(ocr_error_2)
                    else:
                        parsed_komposisi = scan_result_2["parsed"].get("komposisi", "Tidak terdeteksi.")
                        if parsed_komposisi != "Tidak terdeteksi.":
                            sync_ocr_value_to_form("komposisi", parsed_komposisi)
                            clear_ocr_analysis_result()
                            bump_ocr_form_version()

                        st.success("Komposisi berhasil diproses dari satu variasi OCR terbaik agar tidak berulang. Periksa lagi sebelum analisis.")
                        render_ocr_result_debug(scan_result_2, "komposisi")
            except Exception as exc:
                st.error("Gambar komposisi tidak bisa dibaca. Coba upload ulang dalam format JPG atau PNG.")
                with st.expander("Detail error gambar komposisi"):
                    st.code(str(exc))

    st.markdown("---")
    input_col, result_col = st.columns([1.15, 1], gap="large")

    with input_col:
        st.subheader("Konfirmasi Data Input (Hasil OCR)")
        st.warning("Jangan langsung percaya OCR mentah. Koreksi angka dan komposisi sebelum menjalankan rekomendasi.")

        ocr_prefix = f"ocr_{st.session_state.ocr_form_version}"
        product_name, takaran_saji, nutrition_data, komposisi = input_form(ocr_prefix, st.session_state.ocr_data)
        ocr_signature = make_analysis_signature(product_name, takaran_saji, nutrition_data, komposisi)

        if st.button("Analisis dari Data Hasil OCR", type="primary"):
            store_product_analysis_result(
                product_name,
                takaran_saji,
                nutrition_data,
                komposisi,
                store_key="ocr_analysis_result",
                input_signature=ocr_signature,
            )
            st.success("Analisis berhasil diperbarui. Hasil ditampilkan di panel kanan.")

    with result_col:
        st.markdown("### Profil & Analisis Produk Dasar")
        st.caption("Ringkasan prediksi risiko dan kontribusi nutrisi utama.")
        render_analysis_side(st.session_state.ocr_analysis_result, current_signature=ocr_signature)
        
    render_analysis_bottom(st.session_state.ocr_analysis_result, current_threshold)


elif app_mode == "Analisis Batch Excel":
    st.header("Analisis Batch Excel")
    st.write("Upload file Excel dengan kolom Nama Produk, Energi, Lemak, Lemak Jenuh, Karbohidrat, Gula, Protein, Garam, Natrium, Natrium Benzoat, dan Komposisi jika tersedia.")

    uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        
        numeric_cols_to_fix = [
            "Energi", "Lemak", "Lemak Total", "Lemak Jenuh", 
            "Karbohidrat", "Gula", "Protein", "Garam", 
            "Natrium", "Natrium Benzoat", "Takaran Saji", "Takaran"
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
        
        if st.button("Mulai Analisis Batch", type="primary"):
            df_clean = preprocess_batch_excel_data(df)
            results = []
            total_rows = len(df_clean)
            
            progress_bar = st.progress(0)
            
            counter = 0
            for idx, row in df_clean.iterrows():
                nutrition_data = {
                    "energi": row.get("Energi", 0),
                    "lemak_total": row.get("Lemak", row.get("Lemak Total", 0)),
                    "lemak_jenuh": row.get("Lemak Jenuh", 0),
                    "protein": row.get("Protein", 0),
                    "karbohidrat": row.get("Karbohidrat", 0),
                    "gula": row.get("Gula", 0),
                    "garam": row.get("Garam", 0),
                    "natrium": row.get("Natrium", 0),
                    "natrium_benzoat": row.get("Natrium Benzoat", 0),
                }
                komposisi = row.get("Komposisi", "")
                
                takaran_saji = float(row.get("Takaran Saji", row.get("Takaran", 100.0)))
                product_name = str(row.get("Nama Produk", row.get("Produk", row.get("Kemasan", f"Produk {idx+1}"))))

                if not has_sufficient_input(nutrition_data):
                    results.append({
                        "Nama Produk": product_name,
                        "Skor Risiko": "Data belum cukup",
                        "Klasifikasi": "Belum dianalisis",
                        "Rekomendasi": "Isi nilai gizi yang valid sebelum analisis.",
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
                
                counter += 1
                progress_bar.progress(counter / total_rows)
            
            st.session_state.batch_result_df = pd.DataFrame(results)
            st.session_state.batch_total_rows = total_rows
            
        if st.session_state.batch_result_df is not None:
            st.success(f"Analisis batch selesai untuk {st.session_state.batch_total_rows} produk!")
            
            st.header("Hasil Analisis Batch")

            st.markdown("### 🤖 Ringkasan Insight Otomatis")
            insight_text = generate_batch_insights(st.session_state.batch_result_df)
            st.info(insight_text)
            st.markdown("---")
            
            display_cols = ["Nama Produk", "Skor Risiko", "Klasifikasi", "Rekomendasi"]
            st.dataframe(st.session_state.batch_result_df[display_cols], use_container_width=True)
            
            try:
                st.markdown("---")
                st.markdown("### 2. Grafik Distribusi Risiko")

                df_results = st.session_state.batch_result_df.copy()
                
                df_results["Skor Risiko Numerik"] = pd.to_numeric(df_results["Skor Risiko"], errors='coerce')
                valid_df = df_results.dropna(subset=["Skor Risiko Numerik"])

                fig_pie = None

                if not valid_df.empty:
                    classification_colors = {"Aman": "#2ECC71", "Sedang": "#F39C12", "Tinggi": "#E74C3C"}

                    # === PIE CHART ===
                    st.markdown("#### Proporsi Klasifikasi Produk")
                    
                    pie_data = valid_df["Klasifikasi"].value_counts().reset_index()
                    pie_data.columns = ["Klasifikasi", "Jumlah"]
                    pie_colors = [classification_colors.get(c, "#95A5A6") for c in pie_data["Klasifikasi"]]
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=pie_data["Klasifikasi"], 
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
                    st.markdown("#### Peringkat Skor Risiko Produk")
                    
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
                    st.info("Tidak ada data valid yang bisa divisualisasikan dalam grafik.")
                    
                st.markdown("---")
                st.markdown("### 🎯 Detail Pemenuhan Angka Kecukupan Gizi Harian per Produk")
                
                for idx, row in valid_df.iterrows():
                    prod_name = row.get("Nama Produk", "Produk")
                    klasifikasi = row.get("Klasifikasi", "-")
                    skor_num = row.get("Skor Risiko Numerik", 0)
                    
                    with st.expander(f"📦 {prod_name} — Klasifikasi: {klasifikasi} (Skor: {skor_num:.1f}%)"):
                        if 'nutrition_data' in row and isinstance(row['nutrition_data'], dict):
                            nut_data = row['nutrition_data']
                        else:
                            nut_data = {"gula": 0, "natrium": 0, "lemak_jenuh": 0}
                            
                        t_saji = row['takaran_saji'] if 'takaran_saji' in row else 100.0
                        render_health_metrics(nut_data, t_saji, current_threshold, show_header=False)
                
            except Exception as e:
                st.error(f"Terjadi masalah saat merender visualisasi batch: {e}")

            st.markdown("---")
            st.markdown("### 📥 Export Hasil Analisis")

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_to_excel = st.session_state.batch_result_df[display_cols]
                df_to_excel.to_excel(writer, index=False, sheet_name="Hasil Analisis")

            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                st.download_button(
                    label="📊 Download Hasil Excel", 
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
                    with st.spinner("Menyiapkan dokumen PDF Profesional..."):
                        pdf_data = generate_pdf_report(df_results, insight_text, fig_pie)
                        st.download_button(
                            label="📑 Download PDF Report", 
                            data=pdf_data, 
                            file_name="Laporan_Analisis_NutriScan.pdf", 
                            mime="application/pdf",
                            use_container_width=True
                        )
                else:
                    st.warning("⚠️ Modul 'fpdf' belum terinstall. Jalankan `pip install fpdf kaleido` di terminal agar tombol Export PDF muncul.")
            
    else:
        st.session_state.batch_result_df = None


elif app_mode == "Perbandingan Produk":
    st.header("Perbandingan Produk (Food Comparison Mode)")
    st.info("Bandingkan metrik AI (Skor Risiko) dan metrik BI (Kepadatan Energi) dari dua produk sekaligus. Anda dapat menggunakan preset atau OCR untuk memindai label masing-masing produk.")

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

    with colA:
        st.markdown("### 📦 Produk A")
        method_a = st.radio("Metode Input Produk A:", ["Pilih Contoh Produk", "Scan Label (OCR)", "Input Manual"], horizontal=True, key="method_a")
        
        if method_a == "Pilih Contoh Produk":
            st.selectbox("Contoh Uji Produk A", list(EXAMPLE_PRESETS.keys()), key="preset_comp_a", on_change=apply_comp_preset_a)
        elif method_a == "Scan Label (OCR)":
            col_scan_gizi_a, col_scan_komp_a = st.columns(2)
            with col_scan_gizi_a:
                st.markdown("**1. Scan Nilai Gizi**")
                type_a_gizi = st.radio("S. Gizi A", ["Upload", "Kamera"], key="type_a_gizi", horizontal=True, label_visibility="collapsed")
                file_a_gizi = st.file_uploader("Foto Gizi A", type=["jpg", "jpeg", "png"], key="file_a_gizi") if type_a_gizi == "Upload" else st.camera_input("Kamera Gizi A", key="cam_a_gizi")
                if file_a_gizi:
                    img_a_gizi = Image.open(file_a_gizi)
                    safe_image(standardize_image_size(img_a_gizi), caption="Gizi A", width=250)
                    if st.button("🔍 OCR Gizi A", key="btn_ocr_a_gizi"):
                        with st.spinner("Membaca..."):
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
                                    st.success("Tersimpan!")

            with col_scan_komp_a:
                st.markdown("**2. Scan Komposisi**")
                type_a_komp = st.radio("S. Komp A", ["Upload", "Kamera"], key="type_a_komp", horizontal=True, label_visibility="collapsed")
                file_a_komp = st.file_uploader("Foto Komp A", type=["jpg", "jpeg", "png"], key="file_a_komp") if type_a_komp == "Upload" else st.camera_input("Kamera Komp A", key="cam_a_komp")
                if file_a_komp:
                    img_a_komp = Image.open(file_a_komp)
                    safe_image(standardize_image_size(img_a_komp), caption="Komp A", width=250)
                    if st.button("🔍 OCR Komposisi A", key="btn_ocr_a_komp"):
                        with st.spinner("Membaca..."):
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
                                    st.success("Tersimpan!")

        st.markdown("#### Form Data Produk A")
        prod_name_a, saji_a, nut_a, kompo_a = input_form(f"comp_a_{st.session_state.comp_a_ver}", st.session_state.comp_a_data)

    with colB:
        st.markdown("### 📦 Produk B")
        method_b = st.radio("Metode Input Produk B:", ["Pilih Contoh Produk", "Scan Label (OCR)", "Input Manual"], horizontal=True, key="method_b")
        
        if method_b == "Pilih Contoh Produk":
            st.selectbox("Contoh Uji Produk B", list(EXAMPLE_PRESETS.keys()), key="preset_comp_b", on_change=apply_comp_preset_b)
        elif method_b == "Scan Label (OCR)":
            col_scan_gizi_b, col_scan_komp_b = st.columns(2)
            with col_scan_gizi_b:
                st.markdown("**1. Scan Nilai Gizi**")
                type_b_gizi = st.radio("S. Gizi B", ["Upload", "Kamera"], key="type_b_gizi", horizontal=True, label_visibility="collapsed")
                file_b_gizi = st.file_uploader("Foto Gizi B", type=["jpg", "jpeg", "png"], key="file_b_gizi") if type_b_gizi == "Upload" else st.camera_input("Kamera Gizi B", key="cam_b_gizi")
                if file_b_gizi:
                    img_b_gizi = Image.open(file_b_gizi)
                    safe_image(standardize_image_size(img_b_gizi), caption="Gizi B", width=250)
                    if st.button("🔍 OCR Gizi B", key="btn_ocr_b_gizi"):
                        with st.spinner("Membaca..."):
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
                                    st.success("Tersimpan!")

            with col_scan_komp_b:
                st.markdown("**2. Scan Komposisi**")
                type_b_komp = st.radio("S. Komp B", ["Upload", "Kamera"], key="type_b_komp", horizontal=True, label_visibility="collapsed")
                file_b_komp = st.file_uploader("Foto Komp B", type=["jpg", "jpeg", "png"], key="file_b_komp") if type_b_komp == "Upload" else st.camera_input("Kamera Komp B", key="cam_b_komp")
                if file_b_komp:
                    img_b_komp = Image.open(file_b_komp)
                    safe_image(standardize_image_size(img_b_komp), caption="Komp B", width=250)
                    if st.button("🔍 OCR Komposisi B", key="btn_ocr_b_komp"):
                        with st.spinner("Membaca..."):
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
                                    st.success("Tersimpan!")

        st.markdown("#### Form Data Produk B")
        prod_name_b, saji_b, nut_b, kompo_b = input_form(f"comp_b_{st.session_state.comp_b_ver}", st.session_state.comp_b_data)

    st.markdown("---")
    
    if st.button("⚖️ Bandingkan Kedua Produk", type="primary", use_container_width=True):
        res_a = build_analysis_result(prod_name_a, saji_a, nut_a, kompo_a)
        res_b = build_analysis_result(prod_name_b, saji_b, nut_b, kompo_b)

        if res_a.get("status") == "ok" and res_b.get("status") == "ok":
            st.markdown("### 🏆 Kesimpulan Perbandingan AI")
            score_a = res_a["risk_score"]
            score_b = res_b["risk_score"]
            diff = abs(score_a - score_b)

            if score_a < score_b:
                st.success(f"Berdasarkan analisis nutrisi, **{res_a['product_name'] or 'Produk A'}** adalah pilihan yang lebih baik. Skor risikonya **{diff:.2f}% lebih rendah** dibandingkan {res_b['product_name'] or 'Produk B'}.")
            elif score_b < score_a:
                st.success(f"Berdasarkan analisis nutrisi, **{res_b['product_name'] or 'Produk B'}** adalah pilihan yang lebih baik. Skor risikonya **{diff:.2f}% lebih rendah** dibandingkan {res_a['product_name'] or 'Produk A'}.")
            else:
                st.info("Kedua produk memiliki metrik tingkat risiko yang identik secara numerik.")

            fig_comp = go.Figure(data=[
                go.Bar(name=res_a['product_name'] or 'Produk A', x=['Skor Risiko AI (%)'], y=[score_a], marker_color='#3498DB', text=[f"{score_a:.1f}%"], textposition='auto'),
                go.Bar(name=res_b['product_name'] or 'Produk B', x=['Skor Risiko AI (%)'], y=[score_b], marker_color='#E74C3C', text=[f"{score_b:.1f}%"], textposition='auto')
            ])
            fig_comp.update_layout(barmode='group', title="Perbandingan Head-to-Head Skor Risiko", height=400)
            st.plotly_chart(fig_comp, use_container_width=True)

            st.markdown("---")

        row1_colA, row1_colB = st.columns(2, gap="large")
        with row1_colA:
            st.markdown(f"### Hasil: {res_a['product_name'] or 'Produk A'}")
            st.markdown("#### Profil & Analisis Produk Dasar")
            st.caption("Ringkasan prediksi risiko dan kontribusi nutrisi utama.")
            render_analysis_side(res_a)
        with row1_colB:
            st.markdown(f"### Hasil: {res_b['product_name'] or 'Produk B'}")
            st.markdown("#### Profil & Analisis Produk Dasar")
            st.caption("Ringkasan prediksi risiko dan kontribusi nutrisi utama.")
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
            
            st.markdown("### 📊 Profil Gizi & Makronutrien Holistik")
            st.caption("Analisis mendalam mengenai sumber kalori dan dampak glikemik berdasarkan takaran saji.")
            
            row3a_colA, row3a_colB = st.columns(2, gap="large")
            with row3a_colA:
                render_nutrition_kepadatan_gula(res_a["nutrition_data"], res_a["takaran_saji"])
            with row3a_colB:
                render_nutrition_kepadatan_gula(res_b["nutrition_data"], res_b["takaran_saji"])

            st.write("")
            st.markdown("**Distribusi Sumber Kalori (Macronutrient Split)**")
            
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


elif app_mode == "Simulasi Konsumsi Produk":
    st.header("Simulasi Konsumsi Produk")
    st.info("Masukkan detail produk dan perkirakan dampak risikonya berdasarkan frekuensi konsumsi Anda.")

    if "sim_data" not in st.session_state:
        st.session_state.sim_data = init_parsed_data()
    if "sim_ver" not in st.session_state:
        st.session_state.sim_ver = 0

    def apply_sim_preset():
        st.session_state.sim_data = dict(EXAMPLE_PRESETS[st.session_state.preset_sim])
        st.session_state.sim_ver += 1

    st.markdown("### Langkah 1: Definisikan Produk")
    method_sim = st.radio("Metode Input Produk:", ["Pilih Contoh Produk", "Scan Label (OCR)", "Input Manual"], horizontal=True, key="method_sim")
    
    if method_sim == "Pilih Contoh Produk":
        st.selectbox("Contoh Uji Produk", list(EXAMPLE_PRESETS.keys()), key="preset_sim", on_change=apply_sim_preset)
    elif method_sim == "Scan Label (OCR)":
        col_scan_gizi_sim, col_scan_komp_sim = st.columns(2)
        with col_scan_gizi_sim:
            st.markdown("**1. Scan Nilai Gizi**")
            type_sim_gizi = st.radio("S. Gizi Sim", ["Upload", "Kamera"], key="type_sim_gizi", horizontal=True, label_visibility="collapsed")
            file_sim_gizi = st.file_uploader("Foto Gizi Sim", type=["jpg", "jpeg", "png"], key="file_sim_gizi") if type_sim_gizi == "Upload" else st.camera_input("Kamera Gizi Sim", key="cam_sim_gizi")
            if file_sim_gizi:
                img_sim_gizi = Image.open(file_sim_gizi)
                safe_image(standardize_image_size(img_sim_gizi), caption="Gizi Sim", width=250)
                if st.button("🔍 OCR Gizi", key="btn_ocr_sim_gizi"):
                    with st.spinner("Membaca..."):
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
                                st.success("Tersimpan!")

        with col_scan_komp_sim:
            st.markdown("**2. Scan Komposisi**")
            type_sim_komp = st.radio("S. Komp Sim", ["Upload", "Kamera"], key="type_sim_komp", horizontal=True, label_visibility="collapsed")
            file_sim_komp = st.file_uploader("Foto Komp Sim", type=["jpg", "jpeg", "png"], key="file_sim_komp") if type_sim_komp == "Upload" else st.camera_input("Kamera Komp Sim", key="cam_sim_komp")
            if file_sim_komp:
                img_sim_komp = Image.open(file_sim_komp)
                safe_image(standardize_image_size(img_sim_komp), caption="Komp Sim", width=250)
                if st.button("🔍 OCR Komposisi", key="btn_ocr_sim_komp"):
                    with st.spinner("Membaca..."):
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
                                st.success("Tersimpan!")

    prod_name_sim, saji_sim, nut_sim, kompo_sim = input_form(f"sim_{st.session_state.sim_ver}", st.session_state.sim_data)

    st.markdown("---")
    st.markdown("### Langkah 2: Atur Pola Konsumsi")
    col_pola1, col_pola2 = st.columns(2)
    freq_weekly = col_pola1.slider("Frekuensi konsumsi per minggu (kali/sajian)", 1, 21, 3)
    sim_period = col_pola2.slider("Periode Simulasi (Bulan)", 1, 12, 1)

    if st.button("🚀 Jalankan Simulasi", type="primary", use_container_width=True):
        res_sim = build_analysis_result(prod_name_sim, saji_sim, nut_sim, kompo_sim)
        
        if res_sim.get("status") == "ok":
            st.markdown("---")
            st.markdown("### 📈 Hasil Simulasi")
            
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

            st.markdown(f"#### Dampak Akumulatif Selama {sim_period} Bulan (~{total_days} hari)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Sajian Dikonsumsi", f"{int(total_servings)} porsi", f"{freq_weekly}x seminggu")
            c2.metric("Total Gula dari Produk", f"{acc_sugar:.1f} g", f"Setara ~ {acc_sugar/15:.1f} sdm gula")
            c3.metric("Total Kalori dari Produk", f"{acc_cal:.1f} kkal", f"Setara ~ {acc_cal/7700:.1f} kg lemak")

            st.write("")
            st.markdown("##### ⚠️ Persentase Konsumsi Terhadap Batas Maksimal (Angka Kecukupan Gizi) Anda dalam Periode Ini:")
            
            gula_pct = (acc_sugar / max_sugar * 100) if max_sugar else 0
            natrium_pct = (acc_sodium / max_sodium * 100) if max_sodium else 0
            lemak_jenuh_pct = (acc_sat_fat / max_sat_fat * 100) if max_sat_fat else 0
            kalori_pct = (acc_cal / max_cal * 100) if max_cal else 0

            custom_progress_bar("Kalori yang Dihabiskan", acc_cal, max_cal, "kkal", "#10B981", kalori_pct)
            custom_progress_bar("Batas Gula yang Dihabiskan", acc_sugar, max_sugar, "g", "#F59E0B", gula_pct)
            custom_progress_bar("Batas Natrium yang Dihabiskan", acc_sodium, max_sodium, "mg", "#3498DB", natrium_pct)
            custom_progress_bar("Batas Lemak Jenuh yang Dihabiskan", acc_sat_fat, max_sat_fat, "g", "#9B59B6", lemak_jenuh_pct)
            
            st.info("Simulasi di atas menunjukkan seberapa besar jatah nutrisi Anda yang **habis hanya oleh satu jenis produk ini saja** selama periode simulasi. Idealnya, camilan atau minuman tunggal tidak boleh mendominasi batas asupan harian/bulanan Anda.")
            
            st.markdown("---")
            col_simA, col_simB = st.columns(2, gap="large")
            with col_simA:
                st.markdown("### 📋 Profil & Analisis Produk Dasar")
                st.caption("Ringkasan prediksi risiko dan kontribusi nutrisi utama.")
                render_analysis_side(res_sim)
            with col_simB:
                render_holistic_nutrition_profile(res_sim["nutrition_data"], res_sim["takaran_saji"])
                
            st.markdown("---")
            risk_info_sim = res_sim.get("risk_info", classify_risk(res_sim.get("risk_score", 0)))
            render_recommendation_details(risk_info_sim, res_sim.get("recommendation", ""), res_sim.get("is_upf"), res_sim.get("upf_flags", []))
            
            st.markdown("---")
            render_health_metrics(res_sim["nutrition_data"], res_sim["takaran_saji"], current_threshold, show_header=True)
        else:
            st.error("Silakan lengkapi data nutrisi produk untuk menjalankan simulasi.")


elif app_mode == "Riwayat Analisis":
    st.header("Riwayat Analisis")
    if st.session_state.scan_history:
        st.dataframe(pd.DataFrame(st.session_state.scan_history), use_container_width=True)
    else:
        st.info("Belum ada riwayat analisis pada sesi ini.")


elif app_mode == "Edukasi Gizi":
    st.header("Edukasi Gizi")
    st.markdown(
        """
        **Cara membaca hasil aplikasi:**

        1. OCR hanya membantu mengisi data awal, bukan pengganti validasi pengguna.
        2. Data kosong tidak dianalisis agar aplikasi tidak memberi klasifikasi palsu.
        3. Klasifikasi Aman, Sedang, dan Tinggi memakai satu fungsi keputusan.
        4. Gula tinggi perlu diperhatikan karena berpengaruh pada beban asupan harian.
        5. Natrium tinggi perlu dibatasi, terutama pada pengguna dengan risiko hipertensi.
        6. Lemak jenuh tinggi sebaiknya tidak dikonsumsi terlalu sering.
        7. Komposisi dengan pemanis buatan, pewarna sintetik, pengawet, dan penguat rasa menandakan indikasi produk ultra proses.
        """
    )