from flask import Flask, request
from flask.templating import render_template
import requests
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from captcha2str import CaptchaSolver # pylint: disable=import-error


app = Flask(__name__, static_url_path="/static", static_folder="static")
solver = CaptchaSolver(model="data.h5")
# 캡차 이미지 URL
url = "https://sugang.knu.ac.kr/Sugang/captcha"

@app.route("/", methods=['GET', 'POST'])
def main_page():
    [os.remove(f"./static/{f}") for f in os.listdir('./static') if f.startswith('captcha_')]
    filename = f"./static/captcha_{time.time()}.png"
    # 캡차 이미지 다운로드
    with open(f"{filename}", "wb") as captcha_file:
        response = requests.get(url)
        captcha_file.write(response.content)

    result = solver.predict(captcha_img=filename)
    
    return render_template("main.html", captcha=result, filename=filename[9:])

@app.route("/api", methods=["GET", "POST"])
def api():
    if (request.method == 'POST'):
        f = request.files.get("upload_file")
        if (f):
            fname = f.filename
            if not (fname.lower().endswith("png")):
                return "Upload PNG ONLY"
            path = os.path.join("./", fname)
            f.save(path)
            result = solver.predict(captcha_img=path)
            os.remove(path)
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
