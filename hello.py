import streamlit as st
st.title("Hello World!")
st.subheader("This is a simple Streamlit app.")
name = st.text_input("What is your name?")
st.write("Hello, " + name + "!")

