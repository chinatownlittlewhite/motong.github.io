import streamlit as st
import torch
import torchvision.transforms
from PIL import Image
from torch import nn
import os
from DMSHN import DMSHN
jxnu_image_path="images\jxnu.png"
jxnu_img=Image.open(jxnu_image_path)
st.image(jxnu_img)
st.markdown("<h1 style='text-align: center;'>自监督多尺度金字塔融合网络,实现逼真的散景效果渲染效果展示</h1>", unsafe_allow_html=True)
st.write('<p>Zhifeng Wang a, Aiwen Jiang a,*, Chunjie Zhang b, Hanxi Li a, Bo Liu c</p>', unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>简介</h3>", unsafe_allow_html=True)
st.write("<p style='text-align: center;'>**************简介内容************</p>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>散景虚化演示:</h2>", unsafe_allow_html=True)


st.markdown("<h3 style='text-align: center;'>样例图片</h3>", unsafe_allow_html=True)
demo_image_path="images\example.jpg"
demo_image=Image.open(demo_image_path)
st.image(demo_image_path)


st.markdown("<h3 style='text-align: center;'>样例视频</h3>", unsafe_allow_html=True)
demo_video_path="images/Effect_display.mp4"
st.video(demo_video_path)

device = torch.device("cpu")
feed_width = 1536
feed_height = 1024
bokehnet = DMSHN().to(device)
bokehnet.load_state_dict(torch.load('dmshn.pth',map_location=device),False)

st.markdown("<h3 style='text-align: center;'>自定义图片处理</h3>", unsafe_allow_html=True)
The_processed_image_path = st.file_uploader("请上传需要进行虚化的图片", type=["jpg","jpeg","png"])

if The_processed_image_path is not None:
    The_processed_image = Image.open(The_processed_image_path)
    st.write("<p>您上传的图像</p>", unsafe_allow_html=True)
    st.image(The_processed_image)
    The_processed_image=bokehnet(The_processed_image)
    st.write("<p>虚化后的图像</p>", unsafe_allow_html=True)
    st.image(The_processed_image)

