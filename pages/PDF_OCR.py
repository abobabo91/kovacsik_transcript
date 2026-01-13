import streamlit as st
import pandas as pd
import numpy as np
import os
import PyPDF2
from pdf2image import convert_from_bytes
import gc
import pytesseract
from io import BytesIO
import cv2
from PIL import Image

def extract_text_from_pdf(uploaded_file):
    file_name = uploaded_file.name
    pdf_content = ""

    # 1) Sima szövegkinyerés
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            pdf_content += page.extract_text() or ""
    except Exception as e:
        st.error(f"Hiba a(z) {file_name} fájl olvasásakor: {e}")
        return None

    # 2) OCR, ha túl kevés szöveg van
    if len(pdf_content.strip()) < 100:
        pdf_content = ""
        try:
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()

            # determine number of pages
            num_pages = len(PyPDF2.PdfReader(BytesIO(file_bytes)).pages)

            progress = st.progress(0)
            for i in range(1, num_pages + 1):
                # higher DPI for sharper OCR
                images = convert_from_bytes(file_bytes, dpi=300, first_page=i, last_page=i)

                # --- OpenCV preprocessing with Otsu threshold ---
                img = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2GRAY)
                _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # back to PIL for pytesseract
                img_pil = Image.fromarray(img)

                # OCR with English language, PSM 3
                custom_config = r'--psm 3'
                text = pytesseract.image_to_string(img_pil, lang="eng", config=custom_config)
                pdf_content += text + "\n"

                # memóriatisztítás
                del images, img, img_pil
                gc.collect()

                progress.progress(i / num_pages)

        except Exception as e:
            st.error(f"OCR hiba a(z) {file_name} fájlnál: {e}")
            return None

    # 3) hosszkorlátozás
    if len(pdf_content) > 300000:
        st.warning(file_name + " túl hosszú, csak az első 300000 karakter kerül feldolgozásra.")
        pdf_content = pdf_content[:300000]

    return pdf_content

st.set_page_config(page_title="PDF OCR Extractor", layout="wide")
st.title("📄 PDF OCR Extractor")

if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""

col1, col2 = st.columns(2)

with col1:
    st.subheader("PDF feltöltése és OCR")
    uploaded_file = st.file_uploader("Válassz egy PDF fájlt", type=["pdf"])
    if uploaded_file:
        if st.button("Kinyerés indítása"):
            with st.spinner("Feldolgozás..."):
                text = extract_text_from_pdf(uploaded_file)
                if text:
                    st.session_state.extracted_text = text
                    st.success("Szöveg sikeresen kinyerve!")
                else:
                    st.error("Nem sikerült szöveget kinyerni.")

with col2:
    st.subheader("Meglévő szöveg betöltése")
    uploaded_txt = st.file_uploader("Válassz egy korábban mentett .txt fájlt", type=["txt"])
    if uploaded_txt:
        # Read the text file
        try:
            text_content = uploaded_txt.read().decode("utf-8")
            st.session_state.extracted_text = text_content
            st.success("Szöveg sikeresen betöltve!")
        except Exception as e:
            st.error(f"Hiba a szövegfájl beolvasásakor: {e}")

if st.session_state.extracted_text:
    st.divider()
    st.subheader("Feldolgozott szöveg")
    
    # Text area to view/edit the content
    st.session_state.extracted_text = st.text_area(
        "Kinyert/Betöltött szöveg", 
        st.session_state.extracted_text, 
        height=500
    )
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        # Download button
        st.download_button(
            label="Szöveg mentése (.txt)",
            data=st.session_state.extracted_text,
            file_name=f"ocr_output_{pd.Timestamp.now().strftime('%Y%md_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    with c2:
        # Option to use this text in the main app
        if st.button("Használat az interjú transzkripcióhoz"):
            st.session_state.raw_transcription = st.session_state.extracted_text
            st.info("A szöveg átmásolva a főoldalra!")
            
    with c3:
        if st.button("Mező törlése"):
            st.session_state.extracted_text = ""
            st.rerun()
