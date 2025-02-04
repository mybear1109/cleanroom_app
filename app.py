import numpy as np
import streamlit as st  
from PIL import Image, ImageOps
import os
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.layers import DepthwiseConv2D

def main():
    st.title('깨끗한 방인지 더러운 방인지!')
    st.info('방 사진을 업로드 하면, 깨끗한 방인지, 더러운 방인지 알려드립니다.')

    file = st.file_uploader('방 사진을 업로드 해주세요.', type=['jpg', 'jpeg', 'png'])

    if file is not None:
        image = Image.open(file)
        st.image(image)

        model_path = "model/keras_model.h5"

        if not os.path.exists(model_path):
            st.error("❌ 모델 파일을 찾을 수 없습니다. 파일 경로를 확인하세요.")
        else:
            try:
                # ✅ DepthwiseConv2D groups=1 오류 무시
                class CustomDepthwiseConv2D(DepthwiseConv2D):
                    def __init__(self, *args, **kwargs):
                        kwargs.pop("groups", None)  # groups=1 제거
                        super().__init__(*args, **kwargs)

                custom_objects = {"DepthwiseConv2D": CustomDepthwiseConv2D}
                model = load_model(model_path, compile=False, custom_objects=custom_objects)
                
                st.text("✅ Model Loaded Successfully!")
            except Exception as e:
                st.error(f"❌ 모델을 불러오는 중 오류 발생: {e}")
        
        class_names = open("model/labels.txt", "r", encoding='utf-8').readlines()

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # ✅ Pillow 버전에 따라 Resampling 처리 (버전에 따라 다르게 적용)
        size = (224, 224)
        try:
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        except AttributeError:  # Pillow 9.1 미만에서는 Resampling이 없음
            image = ImageOps.fit(image, size, Image.LANCZOS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", confidence_score)

        # Print prediction and confidence score
        st.info(f'이 방은 {class_name[2:]} 방입니다. 확률은 {confidence_score} 정도입니다.')                     


if __name__ == '__main__':
    main()
