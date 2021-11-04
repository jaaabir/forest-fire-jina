import streamlit as st
from PIL import Image
import numpy as np
import os

st.write('''
    # Forest Fire Detection
''')


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

            for img_file in img_files:
                st.write(img_file.name)
                file_details = {"filename": img_file.name, "filetype": img_file.type,
                                "filesize": img_file.size}
                st.write(file_details)
                st.image(load_img(img_file))

                with open(os.path.join("images", img_file.name), "wb") as f:
                    img_buffer = (img_file).getbuffer()
                    f.write(img_buffer)

                st.success("File Saved")

    elif choice == "Dataset":
        st.subheader("Dataset")

    elif choice == "DocumentFiles":
        st.subheader("DocumentFiles")

    elif choice == "About":
        st.subheader("About")

        st.write("This is a project made to detect forest fires using Jina AI. ")


if __name__ == '__main__':
    main()
