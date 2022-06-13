from flask import Flask, render_template, Response, jsonify
import gunicorn
from cam import *

app = Flask(__name__)

headings = ("Name","Album","Artist")
df_1 = music_rec()
df_1 = df_1.head(20)
@app.route('/')
def index():
    print(df_1.to_json(orient='records'))
    return render_template('index.html', headings=headings, data=df_1)

def gen(camera):
    while True:
        global df_1
        frame, df_1 = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/t')
def gen_table():
    return df_1.to_json(orient='records')

if __name__ == '__main__':
    app.debug = True
    app.run()
