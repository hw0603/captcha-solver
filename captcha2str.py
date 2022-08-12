# pyright: reportMissingImports=false
import numpy as np

# 캡차 이미지 픽셀 크기
img_width, img_height = 140, 35
# 캡차 문자열 최대 길이
max_length = 4
# 데이터 처리에 사용할 vocabulary
characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
characters = "".join(sorted(characters))


class CaptchaSolverTF:
    # 생성자
    def __init__(self, model="data.h5"):
        from tensorflow import keras
        import tensorflow as tf
        import os

        self.tf = tf
        self.keras = keras

        # TensorFlow 로그 레벨 설정
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        # CUDA 프로세서 비활성화
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
        # Use "keras.layers.StringLookup" if tensorflow version >=2.4 else "keras.layers.experimental.preprocessing.StringLookup"
        # 문자를 정수형으로 매핑
        self.char_to_num = keras.layers.experimental.preprocessing.StringLookup(vocabulary=list(characters), mask_token=None)
        # 정수를 문자로 매핑
        self.num_to_char = keras.layers.experimental.preprocessing.StringLookup(vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True)
        # Keras 모델 로드
        self.prediction_model = keras.models.load_model(model, compile=False)

    # 이미지 전처리
    def encode_single_sample(self, img_path, raw_bytes=False):
        # 1. 이미지 로드
        img = img_path if raw_bytes else self.tf.io.read_file(img_path)

        # 2. PNG 이미지 디코드 (원본 파일의 채널(RGBA 4채널) 그대로 사용)
        img = self.tf.io.decode_png(img, channels=0)

        # 3. R, G, B, A 채널을 분리한 후 A 채널만 추출하여 그레이스캐일로 변환
        r, g, b, a = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]
        rgb = self.tf.stack([a], axis=-1)
        img = rgb

        # 4. 8bit([0, 255]) 데이터를 float32([0, 1]) 범위로 변환
        img = self.tf.image.convert_image_dtype(img, self.tf.float32)

        # 5. 인식률 향상을 위해 이미지를 이진화(Binarization) 함
        img = self.tf.where(img > 0, 1, 0)

        # 6. 이미지 크기에 맞게 리사이징
        img = self.tf.image.resize(img, [img_height, img_width])
        
        # 7. 이미지 가로세로 바꿈 -> 이미지의 가로와 시간 차원을 대응하기 위함
        img = self.tf.transpose(img, perm=[1, 0, 2])
        
        # 8. 결과 반환
        return img

    # softmax 배열을 문자열로 디코드
    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Greedy Search 사용. 복잡한 작업에서는 Beam Search 사용 가능
        results = self.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
        # CTC Decode 결과에서 문자열 매핑
        output_text = []
        for res in results:
            res = self.tf.strings.reduce_join(self.num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    # 이미지를 인자로 받아서 예측한 문자열 반환
    def predict(self, captcha_img="captcha.png", raw_bytes=False):
        img_data = self.encode_single_sample(captcha_img, raw_bytes=raw_bytes)
        preds = self.prediction_model.predict(self.tf.reshape(img_data, shape=[-1, img_width, img_height, 1]))
        pred_texts = self.decode_batch_predictions(preds)
        
        return pred_texts.pop().replace("[UNK]", "?")


class CaptchaSolverTFLite:
    # 생성자
    def __init__(self, model="data.tflite"):
        # pylint: disable=import-error
        import tflite_runtime.interpreter as tflite
        from PIL import Image
        from itertools import groupby
        import io

        self.Image = Image
        self.groupby = groupby
        self.io = io

        # TFLite 인터프리터 설정
        self.interpreter = tflite.Interpreter(model_path=model)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    # 이미지 전처리
    def preprocess(self, img_path, raw_bytes=False):
        # 0. raw_bytes 가 전달된 경우 BytesIO로 변환
        if (raw_bytes):
            stream = self.io.BytesIO(img_path)
            img = stream
        else:
            img = img_path

        # 1. 이미지 로드
        img = self.Image.open(img)

        # 2. 이미지 크기에 맞게 리사이징
        img = img.resize((img_width, img_height))
        
        # 3. 이미지 디코드 후 그레이스케일 변환
        img = np.array(img)
        r, g, b, a = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]
        rgb = np.stack([a], axis=-1)
        img = rgb

        # 4. 인식률 향상을 위해 이미지를 이진화(Binarization) 함
        img = np.where(img > 0, 255, 0)

        # 5. 8bit([0, 255]) 데이터를 float32([0, 1]) 범위로 변환
        img = np.array(img).astype(np.float32) / 255.0

        # 6. 이미지 가로세로 바꿈 -> 이미지의 가로와 시간 차원을 대응하기 위함
        img = np.transpose(np.reshape(img, [img_height, img_width, 1]), [1, 0, 2])

        # 7. BytesIO 사용한 경우 Stream close
        if (raw_bytes):
            stream.close()

        # 8. 결과 반환
        return img

    # softmax 결과값 후처리
    def postprocess(self, output):
        # softmax 배열들에서 각 최댓값의 인덱스 구함
        max_indices = np.argmax(output[0], axis=1)
        # 최댓값 인덱스의 배열에서의 최댓값을 구분선으로 삼아 의미 있는 데이터 추출
        mylist = [k for k, _ in self.groupby(max_indices) if k != 0]
        mylist = [n if 0 < n <= len(characters) else "?" for n in mylist if n != max(mylist)]

        result = "".join(list(map(lambda x: characters[x-1] if x != "?" else "?", mylist)))

        return result

    # 이미지를 인자로 받아서 예측한 문자열 반환
    def predict(self, captcha_img="captcha.png", raw_bytes=False):
        # 캡챠 이미지 전처리
        image = np.reshape(self.preprocess(captcha_img, raw_bytes=raw_bytes), [-1, img_width, img_height, 1])
        # 인터프리터 Tensor 설정
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        # 인터프리터 호출
        self.interpreter.invoke()
        # 인터프리터에서 결과값 가져옴
        self.output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        # 결과값 후처리 (softmax decode)
        result = self.postprocess(self.output_data)
        
        return result


# 스크립트가 직접 실행되었을 때 테스트 함수 실행
def module_test(dir_path="./captcha_images"):
    import time
    import os

    # TFLite, TensorFlow 중 사용할 모듈 선택
    try:
        import tflite_runtime.interpreter as tflite
    except (ModuleNotFoundError, ImportError):
        print("Using TensorFlow")
        CaptchaSolver = CaptchaSolverTF
    else:
        print("Using TFLite")
        CaptchaSolver = CaptchaSolverTFLite
    finally:
        captchasolver = CaptchaSolver()

    # dir_path의 경로 안에 있는 모든 파일들을 CaptchaSolver로 전달 후 결과 출력
    for (root, directories, files) in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)

            t = time.time()
            result = captchasolver.predict(file_path)
            print(f"{file} -> {result}", end=" => ")
            print(time.time() - t)

            # 추론 결과가 잘못된 경우 오류 메시지 출력
            if ("?" in result):
                print(f"{file} -> {result}")
                print("---> 오류")


if __name__ == "__main__":
    module_test(dir_path="./captcha_images/new_server/labeled_image")
else:
    try:
        import tflite_runtime.interpreter as tflite
    except (ModuleNotFoundError, ImportError):
        print("Using TensorFlow")
        CaptchaSolver = CaptchaSolverTF
    else:
        print("Using TFLite")
        CaptchaSolver = CaptchaSolverTFLite
