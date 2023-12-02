import streamlit as st

# Display the choice options to the user

st.write("Select Your application :)")
option = st.radio("Choose the App", ("Image Match", "Image Search by Text"))

# Function for Image Match App
def Image_Match():
    import streamlit as st
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras.applications.vgg19 import preprocess_input
    from tensorflow.keras.models import Model

    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype': 'truetype'})
    from matplotlib import offsetbox
    import numpy as np
    from tqdm import tqdm

    import glob
    import ntpath
    import cv2

    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn import manifold
    import scipy as sc

    st.title("Multi Modal Retrieval Model ")


    
