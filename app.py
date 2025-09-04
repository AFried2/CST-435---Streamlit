import streamlit as st
import pandas as pd

st.title('CST-435 Streamlit') 
st.text('On a scale from 1-10 how much are you paying attentiion to Artzi?')
st.slider('Attention Level:', 1, 10, 5)