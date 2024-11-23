import streamlit as st
from fastai.vision.all import*

import pathlib
import plotly.express as px
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Rasm yuklash
uploaded_file = st.file_uploader("Rasm yuklang", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Yuklangan rasmni o'qish
    img = PILImage.create(uploaded_file)
    
    # Modelni yuklashA
    learner = load_learner("transport_model.pkl")
    
    # Debugging: Learner turini tekshirish
    st.write(f"Learner turi: {type(learner)}")

    # Rasmni aniqlash
    try:
        # Learner obyektining to'g'ri ekanligini tekshirish
        if isinstance(learner, Learner):
            pred, pred_idx, probs = learner.predict(img)
            
            # Natijani ko'rsatish
            st.image(img, caption='Yuklangan rasm', use_column_width=True)
            st.write(f"Bu rasm: {pred} (Ishonch: {probs[pred_idx]:.2f})")
        else:
            st.error("Learner obyektida xato. Model yuklash jarayonini tekshiring.")
    except Exception as e:
        st.error(f"Rasmni aniqlashda xato: {e}")