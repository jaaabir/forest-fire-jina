from typing import Literal
from jina import DocumentArray, Document, Client
from streamlit import cli as stcli
from about import Contents
import streamlit as st
from PIL import Image
import SessionState
import sys
import os
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import cv2


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
    c = Client(port=55805, protocol='http', host='localhost')
    docs = get_docs(img_names)
    if len(docs) > 0:
        res = c.post(
            on='/search',
            inputs=docs,
            return_results=True,
            on_done= print
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
        fname = os.path.join("images", img)
        os.remove(fname)

    print('Removed the images ...')


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
                    st.error('You removed the images before detecting ...')
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
                    print(img_file)
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
            st.warning(
                'You forgot to remove the images !!! But we did it for you.')
            remove_images(session_state.uploaded_imgs)
            session_state.uploaded_imgs = []

        fire = DocumentArray()
        no_fire = DocumentArray()
        docs =  session_state.docs
        if docs:

            for doc in docs:
                print(f"image class is {doc.docs[0].tags['class']}")
                if doc.docs[0].tags['class'] == '0':
                    no_fire.append(doc.docs[0])
                else:
                    fire.append(doc.docs[0])

        st.markdown('''
        
        ```
        Fire
        ``` 

        ''')

        col1 = st.container()
        col2 = st.container()
        col3 = st.container()

        col1, col2, col3 = st.columns(3)
        cols = {
            'col1': col1,
            'col2': col2,
            'col3': col3
        }

        if fire:
            for i, img_file in enumerate(fire):

                ind = (i % 3)+1

                with cols[f'col{ind}']:
                    st.image(Image.fromarray(img_file.blob))


        st.markdown('---')
        st.markdown('''
        
        ```
        No Fire 
        ```

        ''')

        col1 = st.container()
        col2 = st.container()
        col3 = st.container()

        col1, col2, col3 = st.columns(3)
        cols = {
            'col1': col1,
            'col2': col2,
            'col3': col3
        }

        if no_fire:
            for i, img_file in enumerate(no_fire):

                # ind = (i % 3)+1

                # with cols[f'col{ind}']:
                #     st.image(Image.fromarray(img_file.blob))
                st.image(Image.fromarray(img_file.blob))

    elif choice == "API documention":
        st.subheader("API documention :")
        st.markdown('---')
        st.markdown('**Coming Soon ...**')

    elif choice == "About":
        st.subheader("About : ")
        st.markdown('---')
        st.markdown(Contents, unsafe_allow_html=True)


if __name__ == '__main__':
    if 'images' not in os.listdir():
        os.mkdir('images')

    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())



