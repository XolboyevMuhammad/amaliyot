# import streamlit as st
# from fastai.vision.all import *
# import plotly.express as px
# import pathlib
# from PIL import Image
# import requests
# from io import BytesIO

# # PosixPath moslashuvi
# pathlib.PosixPath = pathlib.Path

# # Sarlavha
# st.markdown("# :rainbow[Tasvirlarni aniqlash]")
# st.write("Klasslar: Car, Airplane, Boat, Fruit, Door, Bicycle, Fish, Bird, Toy, Bear")

# # Rasmni yuklash - fayl yoki link orqali
# st.markdown("> :green[Rasmni ushbu qismga yuklang]")
# file_upload = st.file_uploader("Rasm yuklash (avif, png, jpeg, gif, svg)", type=["avif", "png", "jpeg", "gif", "svg"])
# url_input = st.text_input("Yoki rasmning URL manzilini kiriting")

# # Modelni yuklash
# try:
#     model = load_learner('transport_model.pkl')
# except Exception as e:
#     st.error(f"Modelni yuklashda xatolik: {e}")
#     model = None

# # Rasm yuklash va ko'rsatish
# if file_upload or url_input:
#     try:
#         if file_upload:
#             img = PILImage.create(file_upload)
#             st.image(file_upload, caption="Yuklangan rasm")
#         else:
#             response = requests.get(url_input)
#             img = PILImage.create(BytesIO(response.content))
#             st.image(img, caption="URL orqali yuklangan rasm")
        
#         if isinstance(img, PILImage) and model is not None:
#             pred, pred_id, probs = model.predict(img)
#             st.success(f"Bashorat: {pred}")
#             st.info(f"Ehtimollik: {probs[pred_id] * 100:.1f}%")
            
#             # Diagramma chizish
#             fig = px.bar(x=probs * 100, y=model.dls.vocab, labels={'x': "Ehtimollik (%)", 'y': "Klasslar"}, orientation='h')
#             st.plotly_chart(fig)
#         else:
#             st.error("Tasvirni yoki modelni yuklashda muammo bor.")
#     except Exception as e:
#         st.error(f"Bashorat qilishda xatolik: {e}")

# # Sidebar qo'shimchalar
# st.sidebar.header("Qo'shimcha ma'lumotlar")
# st.sidebar.write("Bizni ijtimoiy tarmoqlarda kuzatib boring:")
# st.sidebar.markdown("[Telegram](https://t.me/M_Xolboyev)")
# st.sidebar.markdown("[Instagram](https://www.instagram.com/muhammad_kholboyev/profilecard/?igsh=MWo5azN2MmM2cGs0aw==)")
# st.sidebar.markdown("[Github](https://github.com/XolboyevMuhammad)")
# st.write("Ushbu dastur Xolboyev Muhammad tomonidan ishlab chiqildi")
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