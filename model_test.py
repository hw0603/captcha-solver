from captcha2str import CaptchaSolver # pylint: disable=import-error
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as Image
import random
import requests
import time
import os
import io


# TensorFlow 로그 레벨 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# CUDA 프로세서 비활성화
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# 테스트 데이터 경로
test_folder = "./testimage"
data_dir = Path(test_folder)


# 모델 데이터 경로
model_path = "./data.h5"


# 인코드/디코드 시 사용할 characters
characters = "kyf3456rhnbpedcgwmx827a"
characters = "".join(sorted(characters))


def getImage(path=test_folder, count=1):
    url = "https://sugang.knu.ac.kr/Sugang/captcha"
    for _ in range(count):
        file_name = f"captcha_{str(time.time()).ljust(18, '0')}.png"
        with open(f"{path}/{file_name}", "wb") as file:
            response = requests.get(url)
            file.write(response.content)

    print(f"캡차 이미지 {count}개 다운로드 완료")


def modelinfo():
    # CaptchaSolver 인스턴스 생성
    solver = CaptchaSolver(model=model_path)
    solver.prediction_model.summary()


def visualize():
    # 모든 이미지의 리스트 구함
    images = sorted(list(map(str, list(data_dir.glob("*.png")))))
    
    # CaptchaSolver 인스턴스 생성
    solver = CaptchaSolver(model=model_path)

    # 랜덤하게 샘플 선택
    images_list = [img for img in random.sample(images, 25)]
    result_list = [solver.predict(captcha_img=img_path) for img_path in images_list]

    # 선택한 샘플 시각화
    _, ax = plt.subplots(5, 5, figsize=(15, 5))
    for i in range(len(result_list)):
        img = Image.imread(images_list[i])
        title = f"Prediction: {result_list[i]}"
        ax[i // 5, i % 5].imshow(img, cmap="gray")
        ax[i // 5, i % 5].set_title(title)
        ax[i // 5, i % 5].axis("off")
    plt.show()


def makeLable(path=test_folder):
    # 모든 이미지의 리스트 구함
    images = sorted(list(map(str, list(data_dir.glob("*.png")))))
    labels = (img.split(os.path.sep)[-1].split(".png")[0] for img in images)

    # CaptchaSolver 인스턴스 생성
    solver = CaptchaSolver(model=model_path)

    # 캡차 해독하여 리스트 생성
    result_list = (solver.predict(captcha_img=img_path) for img_path in images)

    for filename, captcha in zip(labels, result_list):
        src = os.path.join(f"{path}", f"{filename}.png")
        dst = os.path.join(f"{path}", f"{captcha}.png")
        try:
            os.rename(src, dst)
        except FileExistsError:
            print("파일이 이미 존재합니다.")
            os.remove(src)
        print(f"{src} -> {dst}")


def apitest(api_url="127.0.0.1:3000/api"):
    url = "https://sugang.knu.ac.kr/Sugang/captcha"
    try:
        while (True):
            # 캡차 이미지 다운로드
            response = requests.get(url)

            # 스트림에 쓰기
            captcha_file = io.BytesIO()
            captcha_file.write(response.content)
            captcha_file.name = "captcha.png"
            captcha_file.seek(0)

            file = {"upload_file": ("captcha.png", captcha_file.getvalue())}

            t = time.time()
            try:
                result = requests.post(f"http://{api_url}", files=file, timeout=20).text
            except Exception as e:
                result = e

            print(result, end=" => ")
            print(time.time() - t)
            
            if (len(result) != 4 or "?" in result):
                print("^-------------------------------")
                print("[오류발생]")
                print(result)
                print("-------------------------------^")
                with open(f"./{time.time()}_Error.png", "wb") as sf:
                    captcha_file.seek(0)
                    sf.write(captcha_file.read())
            captcha_file.close()
    except KeyboardInterrupt:
        return




# getImage(count=50)
# visualize()
# makeLable()
apitest()
