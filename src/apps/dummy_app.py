import streamlit as st

if __name__ == "__main__":
    text = st.text_area("Text to analyze", "Hello")

    st.write("{} world!".format(text))

