import requests
import time

url = "https://sugang.knu.ac.kr/Sugang/captcha"
test_folder = "testimage"
count = 100

def download(url, file_name):
    with open(f"./{test_folder}/{file_name}", "wb") as file:
        response = requests.get(url)
        file.write(response.content)


for _ in range(count):
    download(url, f"captcha_{str(time.time()).ljust(18, '0')}.png")

print(f"캡차 이미지 {count}개 다운로드 완료")
