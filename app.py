import streamlit as st
import io
from PIL import Image
import numpy as np
import cv2
from model import Generator
from torchvision import transforms
import torch

st.set_page_config(
    page_title="Emotion App!",
    page_icon="😎",
    layout="wide"
)
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

st.markdown("### Правила игры:")
st.markdown("1) Введите сообщение на русском языке. Это может быть что угодно, например, случившееся недавно событие или ваши мысли")
st.markdown("2) Приложение распознает тональность вашего настроения и выведет фото автора приложения с соответствующим настроением")
st.markdown("3) Но нет, это не два разных фото. Это работа генеративной модели! Не верите? Попробуйте загрузить своё фото!)")

st.markdown("Советы по выбору фото: лучше всего брать селфи! Или фотографии, где лицо крупным планом. Чем сильнее фото отличается от этого описания, тем хуже результат.")
# st.markdown("<img width=200px src='https://rozetked.me/images/uploads/dwoilp3BVjlE.jpg'>", unsafe_allow_html=True)
# ^-- можно показывать пользователю текст, картинки, ограниченное подмножество html - всё как в jupyter

text = st.text_area("Введите текст:")
# img =  st.image("testJulifil.jpg")
# ^-- показать текстовое поле. В поле text лежит строка, которая находится там в данный момент
# st.markdown("### Hello, world!")
from transformers import pipeline

trans = pipeline('translation', model = "Helsinki-NLP/opus-mt-ru-en")
classifier = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english")
res = classifier(trans(text)[0]["translation_text"])
# st.markdown(res)
if res[0]['label'] != 'POSITIVE':
    labels = torch.Tensor([[0, 0]])
    # st.markdown("Я сотру улыбку с этого лица!")
else:
    labels = torch.Tensor([[1, 1]])
    # st.markdown("Я сделаю фото улыбчивым!")

# st.markdown(labels)
file = st.file_uploader("Загрузите своё фото:", type=['png','jpeg','jpg'])
if file:

    image_data = file.getvalue()
    # Показ загруженного изображения на Web-странице средствами Streamlit
    # st.image(image_data)
    # Возврат изображения в формате PIL
    image = Image.open(io.BytesIO(image_data))
    # image = Image.open("test"+username+".jpg").convert('RGB')


else:
    image = Image.open("testJulifil.jpg")


transform=transforms.Compose([
    transforms.Resize(64*4),
    transforms.CenterCrop(64*4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
img = transform(image)
img = img.unsqueeze(0)
generator = torch.load("gen_model.pt", map_location=torch.device('cpu'))
# st.write(int(labels.sum()) < 1)
# if int(labels.sum()) > 1:
#     st.markdown("Я сотру улыбку с этого лица!")
# elif int(labels.sum()) < 1:
#     st.markdown("Я сделаю фото улыбчивым!")
x_f2 = generator(img, labels)
res = inv_normalize(x_f2[0]).permute(1,2,0).detach().cpu().numpy()
res = res# *255 #
res = cv2.normalize(res, None, 220, 40, cv2.NORM_MINMAX, cv2.CV_8U)
res = res[:,:,::-1]

# plt.imshow(res)
# cv2.imshow("res", res)
username = "0"
cv2.imwrite("res"+username+".jpg", res)
img = st.image(r"res"+username+".jpg")
# x = np.array(img)
# cv2.imwrite("downloaded.jpeg", x)
# st.write(x.mean())
# import cv2
# foto = cv2.imread(file)
# cv2.imwrite(foto, "res.png")
# выводим результаты модели в текстовое поле, на потеху пользователю