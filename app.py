from flask import Flask, render_template, request, Response
import cv2
import os
import numpy as np
import acapture

app = Flask(__name__)
camera = cv2.VideoCapture(-1)
# camera = acapture.open(0)
threshold_value = 100
rainbow_mode = False
hue = 0

@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')


@app.route('/about', methods=["GET"])
def about():
    return render_template('about.html')

@app.route('/convert', methods=['GET', 'POST'])
def convert():
    global threshold_value
    global rainbow_mode
    if request.method =='POST':
        action= request.form["threshhold"]
        if action == 'increase':  
            threshold_value = threshold_value + 10
            print(threshold_value)
            if threshold_value > 255:                    
                threshold_value = 255
        elif action == 'decrease':
            threshold_value = threshold_value - 10
            print(threshold_value)
            if threshold_value < 0:
                threshold_value = 0
        elif action == 'rb':
            rainbow_mode = not rainbow_mode

    gen_frames(threshold_value, rainbow_mode)
    return render_template('convert.html', threshold=threshold_value)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(threshold_value, rainbow_mode), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/uploads', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'photo' in request.files:
            photo = request.files['photo']
            filename = photo.filename
            file_path = os.path.join(app.root_path, 'static/result/og.jpg')
            photo.save(file_path)

            detection_option1 = request.form.get('Detection Option1')

            if detection_option1 == 'Contour':
                edges_path1 = contour_edges(file_path)
            elif detection_option1 == 'Sobel':
                edges_path1 = sobel_edges(file_path)
            elif detection_option1 == 'Canny':
                edges_path1 = canny_edges(file_path)

            detection_option2 = request.form.get('Detection Option2')

            if detection_option2 == 'Contour':
                edges_path2 = contour_edges(file_path)
            elif detection_option2 == 'Sobel':
                edges_path2 = sobel_edges(file_path)
            elif detection_option2 == 'Canny':
                edges_path2 = canny_edges(file_path)

            img1 = cv2.imread(edges_path1)
            img2 = cv2.imread(edges_path2)
            
            dst = cv2.addWeighted(img1, 1, img2, 1, 0)

            edges_path = os.path.join(app.root_path, 'static/result/blended.jpg')
            # cv2.imwrite(edges_path, dst)
            
            # edges_path = os.path.join(app.root_path, 'static/result/inverted.jpg')
            # img = cv2.imread(edges_path)
            cv2.imwrite(edges_path, dst)

            return render_template('results.html', ed1 = detection_option1, ed2 = detection_option2)
        return 'No photo uploaded.'

def canny_edges(image_path):
    image = cv2.imread(image_path, 0)
    img_blur = cv2.GaussianBlur(image, (5,5), 0)
    edges = cv2.Canny(img_blur, 140, 170)
    img_not = cv2.bitwise_not(edges)
    edges_path = os.path.join(app.root_path, 'static/result/Canny.jpg')
    cv2.imwrite(edges_path, img_not)
    return edges_path

def sobel_edges(image_path):
    image = cv2.imread(image_path, 0)
    img_blur = cv2.GaussianBlur(image, (5,5), 0)
    edges = cv2.Sobel(img_blur, cv2.CV_32F, 1, 1, ksize=5)
    edges_path = os.path.join(app.root_path, 'static/result/Sobel.jpg')
    cv2.imwrite(edges_path, edges)
    img2 = cv2.imread(edges_path)
    img_not = cv2.bitwise_not(img2)
    cv2.imwrite(edges_path, img_not)
    return edges_path

def contour_edges(image_path):
    image = cv2.imread(image_path, 0)
    ret, image_threshold = cv2.threshold(image, 140, 255, 0)
    contours, hierarchy = cv2.findContours(image_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape
    image_outlines_only = np.zeros((height, width), np.uint8)
    # image_color = (0, 0, 0)
    # image_outlines_only[:] = image_color
    edges = cv2.drawContours(image_outlines_only, contours, -1, (255, 255, 255), 1)
    img_not = cv2.bitwise_not(edges)
    edges_path = os.path.join(app.root_path, 'static/result/Contour.jpg')
    cv2.imwrite(edges_path, img_not)
    return edges_path

def gen_frames(threshold_value, rainbow_mode):  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break

        else:
            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, image_threshold = cv2.threshold(image_gray, threshold_value, 255, 0)
            contours, hierarchy = cv2.findContours(image_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            height, width, channels = frame.shape
            image_outlines = np.zeros((height,width,channels), np.uint8)
            image_color = (255,255,255)
            image_outlines[:] = image_color
            outline_thickness = 2
            outline_color = (0,0,0)
            contour_id = -1

            if rainbow_mode:
                outline_color = get_next_color()
                image_color = (0,0,0)
                image_outlines[:] = image_color
            else:
                outline_color = (0,0,0)
            
            frame_edges = cv2.drawContours(image_outlines, contours, contour_id, outline_color, outline_thickness)
            ret, buffer = cv2.imencode('.jpg', frame_edges)
            frame = buffer.tobytes()
           
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def get_next_color():  
    # https://en.wikipedia.org/wiki/HSL_and_HSV#HSV_to_RGB_alternative
    global hue
    hue = hue + 10
    r = HSV_to_RGB_aux(5) * 255
    g = HSV_to_RGB_aux(3) * 255
    b = HSV_to_RGB_aux(1) * 255

    return (r,g,b)


def HSV_to_RGB_aux(n):
    s = 1
    v = 1
    k = (n + hue / 60) % 6
    return v - v*s*max(0 , min(k, 4 - k, 1)) 

if __name__ == '___main__':
    app.run(debug= True)
