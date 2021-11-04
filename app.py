import streamlit as st
from PIL import Image
import numpy as np
import os
from jina import DocumentArray, Document, Client
import re
import sys
from streamlit import cli as stcli

st.write('''
    # Forest Fire Detection
''')


def get_docs(img_names) -> DocumentArray:
    folder = 'images'
    images = [Document(uri = f"{folder}/{i}") for i in img_names]

    docs = DocumentArray()
    for doc in images:
        doc.convert_image_uri_to_blob()
        docs.append(doc) 

    return docs

def send_request(img_names):
    c = Client(port = 12345, protocol = 'http', host = 'localhost')
    res = c.post(
        on = '/search',
        inputs = get_docs(img_names),
        return_results = True,
        on_done = print
    )

def load_img(img_file):
    img = Image.open(img_file)
    return img


def main():

    menu = ["Upload Images", "Dataset", "DocumentFiles", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Upload Images":
        st.subheader("Upload Images")
        img_files = st.file_uploader("Select Images",
                                     type=["png", "jpg", "jpeg"], accept_multiple_files=True)

        if img_files:
            img_names = []
            for img_file in img_files:
                img_names.append(img_file.name)
                file_details = {"filename": img_file.name, "filetype": img_file.type,
                                "filesize": img_file.size}

                # st.write(file_details)
                # st.write(img_file.name)
                st.image(load_img(img_file))

                with open(os.path.join("images", img_file.name), "wb") as f:
                    img_buffer = (img_file).getbuffer()
                    f.write(img_buffer)

            if st.button('Detect Fire'):
                send_request(img_names)
            st.success("File Saved")

    elif choice == "Dataset":
        st.subheader("Dataset")

    elif choice == "DocumentFiles":
        st.subheader("DocumentFiles")

    elif choice == "About":
        st.subheader("About")

        st.write("This is a project made to detect forest fires using Jina AI. ")


if __name__ == '__main__':
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())