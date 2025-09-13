import streamlit as st
import pandas as pd

st.title('CST-435 Streamlit') 
st.text('On a scale from 1-10 how much are you paying attentiion to Artzi?')
st.slider('Attention Level:', 1, 10, 5)
st.text('What is your favorite programming language?')
st.selectbox('Programming Language:', ['Python', 'JavaScript', 'Java', 'C++', 'Other'])
st.text('What is your favorite web framework?')
st.selectbox('Web Framework:', ['Streamlit', 'Django', 'Flask', 'React', 'Other'])
