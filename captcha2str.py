from tensorflow import keras
import tensorflow as tf
import numpy as np
import os


class CaptchaSolver:
    # TensorFlow 로그 레벨 설정
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # CUDA 프로세서 비활성화
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # Keras batch_size 설정
    batch_size = 1
    # 캡차 이미지 픽셀 크기
    img_width, img_height = 150, 40
    # 캡차 문자열 최대 길이
    max_length = 4
    # 데이터 처리에 사용할 vocabulary
    characters = "kyf3456rhnbpedcgwmx827a"
    
    # Use "keras.layers.StringLookup" if tensorflow version >=2.4 else "keras.layers.experimental.preprocessing.StringLookup"
    # 문자를 정수형으로 매핑
    char_to_num = keras.layers.experimental.preprocessing.StringLookup(vocabulary=list(characters), mask_token=None)
    # 정수를 문자로 매핑
    num_to_char = keras.layers.experimental.preprocessing.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

    # 생성자
    def __init__(self, model="data.h5"):
        # Keras 모델 로드
        self.prediction_model = keras.models.load_model(model, compile=False)

    # 이미지 전처리
    def encode_single_sample(self, img_path):
        # 1. 이미지 로드
        img = tf.io.read_file(img_path)
        # 2. PNG 이미지 디코드 이후 그레이스케일로 변환
        img = tf.io.decode_png(img, channels=1)
        # 3. 8bit([0, 255]) 데이터를 float32([0, 1]) 범위로 변환
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. 이미지 크기에 맞게 리사이징
        img = tf.image.resize(img, [self.img_height, self.img_width])
        # 5. 이미지 가로세로 바꿈 -> 이미지의 가로와 시간 차원을 대응하기 위함
        img = tf.transpose(img, perm=[1, 0, 2])
        # 6. 결과 반환
        return img

    # softmax 배열을 문자열로 디코드
    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Greedy Search 사용. 복잡한 작업에서는 Beam Search 사용 가능
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :self.max_length]
        # CTC Decode 결과에서 문자열 매핑
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    # 이미지를 인자로 받아서 예측한 문자열 반환
    def predict(self, captcha_img="captcha.png"):
        img_data = self.encode_single_sample(captcha_img)
        preds = self.prediction_model.predict(tf.reshape(img_data, shape=[-1, 150, 40, 1]))
        pred_texts = self.decode_batch_predictions(preds)
        
        return pred_texts.pop().replace("[UNK]", "?")


# 스크립트가 직접 실행되었을 때 테스트 함수 실행
def module_test(filename="captcha.png"):
    import requests
    import time
    # 캡차 이미지 URL
    url = "https://sugang.knu.ac.kr/Sugang/captcha"

    captchasolver = CaptchaSolver(model="data.h5")

    while (True):
        # 캡차 이미지 다운로드
        with open(f"./{filename}", "wb") as captcha_file:
            response = requests.get(url)
            captcha_file.write(response.content)

        t = time.time()
        print(captchasolver.predict(filename), end=" => ")
        print(time.time() - t)

        try:
            input()
        except KeyboardInterrupt:
            break



if __name__ == "__main__":
    module_test(filename="captcha.png")
