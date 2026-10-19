"""Image Enhancer v2 — AI photo enhancer (Streamlit, free hosting).

Real-ESRGAN AI upscaling (4x) + gentle cleanup. Roman Urdu + English UI.

Deploy: upload these files to a GitHub repo -> share.streamlit.io -> New app -> Deploy.
"""
import io
import time

import streamlit as st
from PIL import Image

from enhance import enhance, MAX_OUT

st.set_page_config(page_title='Image Enhancer v2', page_icon='✨')
st.title('✨ Image Enhancer v2')
st.caption('AI photo enhance · 4x bara + saaf · Free · Browser mein chalta hai')

up = st.file_uploader('Apni photo upload karo / Upload your photo',
                      type=['jpg', 'jpeg', 'png', 'webp'])

col1, col2 = st.columns(2)
with col1:
    do_sharpen = st.checkbox('Halki sharpening (subtle)', value=True,
                             help='Bohat halki — koi white halos nahi.')
with col2:
    do_contrast = st.checkbox('Auto-contrast', value=False,
                              help='Default OFF — noise barha sakta hai.')

if up:
    orig = Image.open(up).convert('RGB')
    st.image(orig, caption=f'Original: {orig.width}x{orig.height}', width=350)

    if st.button('Enhance karo ✨', type='primary'):
        with st.spinner('AI enhance ho raha hai... (bari photo par 1-3 min lag sakta hai)'):
            t0 = time.time()
            try:
                result, note = enhance(orig, do_sharpen=do_sharpen,
                                       do_contrast=do_contrast)
            except Exception as e:  # noqa: BLE001 - surface plainly
                st.error(f'Enhance failed: {e}. Dobara try karo.')
            else:
                dt = time.time() - t0
                if note:
                    st.info(note)
                st.image(result,
                         caption=f'Enhanced: {result.width}x{result.height} ({dt:.0f}s)',
                         width=350)
                buf = io.BytesIO()
                fmt = st.radio('Download format', ['PNG', 'JPG'], horizontal=True)
                if fmt == 'PNG':
                    result.save(buf, format='PNG')
                    mime = 'image/png'
                else:
                    result.save(buf, format='JPEG', quality=92)
                    mime = 'image/jpeg'
                st.download_button('⬇️ Download', data=buf.getvalue(),
                                   file_name=f'enhanced.{fmt.lower()}', mime=mime)
                st.success('Ho gaya! ✅')

st.divider()
st.caption(f'Honest note: AI 4x tak asli detail banata hai. Output {MAX_OUT}px tak. '
           '16K jaisay daway jhoot hotay hain — yeh real enhancer hai.')
