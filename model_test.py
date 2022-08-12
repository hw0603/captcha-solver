from cgi import test
from captcha2str import CaptchaSolver # pylint: disable=import-error
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as Image
import random
import requests
import time
import os


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
characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
characters = "".join(sorted(characters))


# [Deprecated] 서버에서 캡차 이미지 다운로드
def getImage(path=test_folder, count=1):
    raise Exception(f"Deprecated function. Please save images to '{test_folder}' manually.")

    url = "https://some-test-url/"
    for _ in range(count):
        file_name = f"captcha_{str(time.time()).ljust(18, '0')}.png"
        with open(f"{path}/{file_name}", "wb") as file:
            response = requests.get(url)
            file.write(response.content)

    print(f"캡차 이미지 {count}개 다운로드 완료")


# 예측 모델의 정보 출력
def modelinfo():
    # CaptchaSolver 인스턴스 생성
    solver = CaptchaSolver(model=model_path)
    solver.prediction_model.summary()


# 예측 결과 시각화
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


# test_folder 내의 모든 이미지 파일명을 예측 결과값으로 변경
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


# Captcha Solver API 테스트
def apitest(api_url="127.0.0.1:3000/api"):
    try:
        for (root, directories, files) in os.walk(test_folder):
            for f in files:
                file_path = os.path.join(root, f)

                with open(file_path, "rb") as captcha_file:
                    file = {"upload_file": ("captcha.png", captcha_file)}

                    t = time.time()
                    try:
                        result = requests.post(f"http://{api_url}", files=file, timeout=20).text
                    except Exception as e:
                        result = e

                    print(f"{f} -> {result}", end=" => ")
                    print(time.time() - t)
                    
                    if (len(result) != 4 or "?" in result):
                        print("^-------------------------------")
                        print("[오류발생]")
                        print(result)
                        print("-------------------------------^")
                        with open(f"./{time.time()}_Error.png", "wb") as sf:
                            captcha_file.seek(0)
                            sf.write(captcha_file.read())
        
    except KeyboardInterrupt:
        return


# modelinfo()
# visualize()
# makeLable()
apitest()
