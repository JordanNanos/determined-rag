import streamlit as st
import numpy as np



st.title("RAG Demo for Financial Services Q&A")

st.markdown("What's your question?")
user_input = st.text_input(label="user input", key="input", label_visibility="hidden")

#col1, col2 = st.columns([3, 1])
# col1, col2 = st.columns(2)
# data = np.random.randn(10, 1)

# col1.subheader("Fine tuned model")
# col1.line_chart(data)

# col2.subheader("Base model")
# col2.write(data)


col1, col2 = st.columns(2)
with col1:
    st.markdown("Fine Tuned Model output")
    st.markdown("**Related Documents via embeddings from fine tuned model**")

with col2:
    st.markdown("Base Model output")
    st.markdown("**Related Documents via embeddings from fine tuned model**")


# structure

# text box

# 2 columns  (fine tuned vs baseline)

# each column
# header
# text box - model output
# text box - Source of Truth

# scroll bars

# highlight???


