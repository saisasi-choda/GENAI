import streamlit as st
from rag import new_doc_pipeline,fetch_relevant_info,get_indices
import os

st.set_page_config(page_title="LIC  Genie", layout="centered")
st.image("./download.jpeg")
st.title("LIC Genie")
uploaded_file = st.sidebar.file_uploader("Upload pdf or csv", type=["csv","pdf"])
st.sidebar.header("Upload Your File")
if uploaded_file:
    file_path = f"./data/{uploaded_file.name}"
    filename = uploaded_file.name
    name, extension = os.path.splitext(filename)
    index_name =name
    # Save the uploaded file locally
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.write(f"File saved sucessfully")
    resp=new_doc_pipeline(file_path=file_path,index_name=index_name)
st.sidebar.title("Available docs")
indx=get_indices()
curr_idx = st.sidebar.selectbox("Select a document:",indx)
# Chat Section
st.subheader(" Upload your policy documents and get a instant answers regarding your policy")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input ("Ask a question based on the PDFs:")

if curr_idx==None:
    st.warning("Please upload your LIC documents")
    
if query and curr_idx:
    response = fetch_relevant_info(query,curr_idx)
    # Append to chat history
    st.session_state.chat_history.append(("You", query))
    st.session_state.chat_history.append(("AI", response))

# Display chat history
for role, text in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧑 {role}:** {text}")
    else:
        st.markdown(f"**🤖 {role}:** {text}")       
