import numpy as np
import streamlit as st
import torch
import torchvision
import PIL.Image as pil
import torch.nn.functional as F
from torch import nn
import os
from DMSHN import DMSHN
jxnu_image_path="./images/jxnu.png"
jxnu_img=pil.open(jxnu_image_path)
st.image(jxnu_img)
st.markdown("<h1 style='text-align: center;'>自监督多尺度金字塔融合网络,实现逼真的散景效果渲染效果展示</h1>", unsafe_allow_html=True)
st.write('<p>Zhifeng Wang a, Aiwen Jiang a,*, Chunjie Zhang b, Hanxi Li a, Bo Liu c</p>', unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>简介</h3>", unsafe_allow_html=True)
st.write("<p style='text-align: center;'>**************简介内容************</p>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>散景虚化演示:</h2>", unsafe_allow_html=True)


st.markdown("<h3 style='text-align: center;'>样例图片</h3>", unsafe_allow_html=True)
demo_image_path="./images/example.jpg"
demo_image=pil.open(demo_image_path)
st.image(demo_image_path)


st.markdown("<h3 style='text-align: center;'>样例视频</h3>", unsafe_allow_html=True)
demo_video_path="./images/Effect_display.mp4"
st.video(demo_video_path)

device = torch.device("cpu")
feed_width = 1536
feed_height = 1024
bokehnet = DMSHN().to(device)
bokehnet.load_state_dict(torch.load('dmshn.pth',map_location=device),False)

st.markdown("<h3 style='text-align: center;'>自定义图片处理</h3>", unsafe_allow_html=True)
The_processed_image_path = st.file_uploader("请上传需要进行虚化的图片", type=["jpg","jpeg","png"])

if The_processed_image_path is not None:
    The_processed_image = pil.open(The_processed_image_path)
    st.write("<p>您上传的图像</p>", unsafe_allow_html=True)
    st.image(The_processed_image)
    with torch.no_grad():
        # Load image and preprocess
        input_image = pil.open(The_processed_image_path).convert('RGB')
        original_width, original_height = input_image.size
        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = torchvision.transforms.ToTensor()(input_image).unsqueeze(0)
        input_image = input_image.to(device)
        bok_pred = bokehnet(input_image)
        bok_pred = F.interpolate(bok_pred, (original_height, original_width), mode='bilinear')
    if bok_pred is not None:
        file_name = bok_pred.name
        file_path = os.path.join("images", file_name)
        with open(file_path, "wb") as f:
            f.write(image_file.getbuffer())
        with open(file_path, "rb") as f:
            content = base64.b64encode(f.read()).decode("utf-8")
            repo.create_file(file_path, f"Added {file_name}", content)
    st.write("<p>虚化后的图像</p>", unsafe_allow_html=True)
    bok_pred = pil.open(file_path)
    st.image(bok_pred)

