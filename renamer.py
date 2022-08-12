"""
캡차 이미지 다운로드 후, 학습 데이터를 생성할 때 라벨링을 편하게 하기 위한 도구
"""

from tkinter import Tk, ttk, StringVar, Label
import PIL.ImageTk
import PIL.Image
import sys
import os


# 캡차 이미지 보여주는 GUI 창
def showCaptcha(captcha_file):
    root = Tk()
    root.resizable(width=False, height=False)
    root.title("Captcha image file renamer")

    def clicked(keypress_event=None):
        root.destroy()

    w = 500 # Tk 창 너비
    h = 200 # Tk 창 높이
    ws = root.winfo_screenwidth() # 화면 너비
    hs = root.winfo_screenheight() # 화면 높이
    x = int((ws/2)-(w/2))
    y = int((hs/2)-(h/2))
    # Tk 창이 화면 중앙에 위치하도록 좌표 설정
    root.geometry(f"{w}x{h}+{x}+{y}")

    # 캡차 파일명 레이블
    label = ttk.Label(root, text=f"{captcha_file.split('//')[-1]}")
    label.pack(side="top")
    # 자동입력 방지 문자 이미지 할당
    img = PIL.ImageTk.PhotoImage(PIL.Image.open(captcha_file).resize((280, 70)))
    panel = Label(root, image=img)
    panel.pack(side="top")
    # 입력 버튼
    action = ttk.Button(root, text="확인", command=clicked)
    action.pack(side="bottom")
    # 입력 창
    captcha = StringVar()
    textbox = ttk.Entry(root, width=20, textvariable=captcha, justify="center", font="Helvetica 44 bold")
    textbox.pack(side="bottom")
    textbox.bind("<Return>", clicked)
    textbox.focus_set()
    # Tk 창 포커싱
    root.focus_force()
    # Tk 창 활성화
    root.mainloop()

    return captcha.get()


# 라벨링 할 파일들이 들어있는 폴더의 경로 설정
target_dir_path = "./testimage"

# target_dir_path 안의 모든 파일들을 순회하며 GUI 창을 띄우고, 새 이름을 입력받고 파일명 변경
for (root, directories, files) in os.walk(target_dir_path):
    for file in files:
        file_path = os.path.join(root, file)
        while (True):
            new_filename = showCaptcha(file_path)
            # exit 입력하면 프로그램 종료
            if (new_filename.lower() == "exit"):
                sys.exit(0)
            
            if (len(new_filename) == 4):
                prev_filename = new_filename
                break

            # 실수로 잘못 입력한 경우 다음에 나오는 창에 "11111" 을 입력하면 잘못 입력한 파일명을 출력
            if (new_filename.startswith("11111")):
                print(prev_filename)
        
        # 파일명 변경
        os.rename(file_path, f"{target_dir_path}//{new_filename.upper()}.png")
