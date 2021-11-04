import enum
import streamlit as st
from PIL import Image
import numpy as np
import os
from jina import DocumentArray, Document, Client
import re
import sys
from streamlit import cli as stcli
import SessionState
# from test import remove_images

st.write('''
    # Forest Fire Detection
''')


def get_docs(img_names) -> DocumentArray:
    folder = 'images'
    try:
        images = [Document(uri=f"{folder}/{i}") for i in img_names]
        docs = DocumentArray()

        for doc in images:
            doc.convert_uri_to_image_blob()
            docs.append(doc)
    except:
        docs = DocumentArray()

    return docs


def send_request(img_names) -> DocumentArray:
    c = Client(port=12344, protocol='http', host='localhost')
    docs = get_docs(img_names)
    if len(docs) > 0:
        res = c.post(
            on='/search',
            inputs=docs,
            return_results=True,
            on_done=print
        )
    else:
        res = -1
    return res


def load_img(img_file):
    img = Image.open(img_file)
    return img


def remove_images(img_names=None):
    folder = 'images'
    img_names = os.listdir(folder) if not img_names else img_names
    for img in img_names:
        fname = f'{folder}/{img}'
        os.remove(fname)

    print('removed the images ...')


def main():

    menu = ["Upload Images", "Predicted Images", "API documention", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    session_state = SessionState.get(
        docs=None, remove_imgs=False, uploaded_imgs=[])

    if choice == "Upload Images":
        st.subheader('Upload Images : ')
        st.markdown('---')
        img_files = st.file_uploader("Select Images",
                                     type=["png", "jpg", "jpeg"], accept_multiple_files=True,)

        if len(img_files) > 0:
            if st.button('Detect Fire'):
                my_bar = st.progress(0)
                session_state.uploaded_imgs = [i.name for i in img_files]
                my_bar.progress(25)
                session_state.docs = send_request(session_state.uploaded_imgs)
                if session_state.docs == -1:
                    st.error('you removed the imaged before detecting ...')
                else:
                    st.markdown('''
                    <style>
                    .pi {
                        font-size: 20px;
                        color : #FF4B4B;
                        font-family: sans-serif; 
                    }
                    </style>
                    <p class='pi'>Go to Predicted Images section</p>
                    ''', unsafe_allow_html=True)
                my_bar.progress(100)
            if st.button('Remove Images'):
                session_state.remove_imgs = True

        col1 = st.container()
        col2 = st.container()
        col3 = st.container()

        col1, col2, col3 = st.columns(3)
        cols = {
            'col1': col1,
            'col2': col2,
            'col3': col3
        }

        if img_files:
            for i, img_file in enumerate(img_files):

                ind = (i % 3)+1

                with cols[f'col{ind}']:
                    st.image(load_img(img_file))

                with open(os.path.join("images", img_file.name), "wb") as f:
                    img_buffer = (img_file).getbuffer()
                    f.write(img_buffer)

            st.success("File Saved")

        if session_state.remove_imgs:
            remove_images(session_state.uploaded_imgs)
            session_state.uploaded_imgs = []
            session_state.remove_imgs = False

    elif choice == "Predicted Images":
        st.subheader("Predicted Images :")
        st.markdown('---')
        if len(session_state.uploaded_imgs) > 0:
            st.warning('you forgot to remove the images !!!')
            remove_images(session_state.uploaded_imgs)

        st.write(session_state.docs)

    elif choice == "API documention":
        st.subheader("API documention :")
        st.markdown('---')

    elif choice == "About":
        st.subheader("About : ")
        st.markdown('---')

        st.markdown('''
        ##### This is a project made to detect forest fires using Jina AI.
        ''')


if __name__ == '__main__':
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
