import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import time
import pandas as pd
import numpy as np
from ultralytics import YOLO

# Cargar modelo YOLO

@st.cache_data
def load_model_sa():
	  return YOLO("best_model_sa_1.pt")

@st.cache_data
def load_model_ca():
	  return YOLO("best_proyecto_conaumentacion.pt")


# Cargar modelo YOLO
#model_proyecto=load_model_sa()  #este es el modelo entrenado sin datos aumentados.
#model_proyecto_aug=YOLO("best_proyecto_v11_6.pt") # este es el modelo entrenado con datos aumentados
model_proyecto_aug=load_model_ca() # este es el modelo entrenado con datos aumentados

st.text("seleccione una imagen y presione el botón de 'Procesar' para detectar las plantas de papa ")