from flask import Flask, request
from flask.templating import render_template
import threading
import requests
import base64
import sys
import os
import io

currentdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(currentdir))
from captcha2str import CaptchaSolver # pylint: disable=import-error


app = Flask(__name__, static_url_path="/static", static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB 업로드 제한

solver = CaptchaSolver()
lock = threading.Lock()

# 캡차 이미지 URL
url = "https://sugang.knu.ac.kr/Sugang/captcha"

@app.route("/", methods=['GET', 'POST'])
def main_page():
    # 캡차 이미지 다운로드
    response = requests.get(url)

    # 파일에 쓰기
    captcha_file = io.BytesIO()
    captcha_file.write(response.content)
    captcha_file.seek(0)

    # BytesIO 인코드/디코드
    encoded_img = base64.b64encode(captcha_file.getvalue())
    decoded_img = encoded_img.decode('utf-8')
    img_data = f"data:image/png;base64,{decoded_img}"
    captcha_file.close()

    with lock:
        result = solver.predict(captcha_img=response.content, raw_bytes=True)
    
    return render_template("main.html", captcha=result, img_data=img_data)

@app.route("/api", methods=["GET", "POST"])
def api():
    if (request.method == 'POST'):
        f = request.files.get("upload_file")
        if (request.form.get("checkonline")):
            return "success"
        if (f):
            fname = f.filename
            if not (fname.lower().endswith("png")):
                return "Accept PNG ONLY"
            with lock:
                result = solver.predict(captcha_img=f.read(), raw_bytes=True)
            return result
        else:
            return "File Not Exist"
    else:
        return render_template("api.html")


if __name__ == "__main__":
    try:
        PORT = 3000
        app.run(os.getenv('FS_BIND', '0.0.0.0'), PORT, threaded=True, debug=True)
    except OSError as e:
        print(f"{PORT}번 포트가 이미 사용 중입니다.\n", e)
        os._exit(1)
