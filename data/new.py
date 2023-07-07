#Import necessary libraries
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import base64

# import cv2
import numpy as np
from mediapipe import Image
import time
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.9)
mpDraw = mp.solutions.drawing_utils
prev_timestamp = 0
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode
className = ''
# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    if result.gestures and result.gestures[0]:
        gesture = result.gestures[0][0]  # Access the first gesture category
        category_name = gesture.category_name
        # print('Gesture recognition result:', category_name)
        global className
        className = category_name
        # global frame
        # cv2.putText(frame, category_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
        #                 1, (0,0,255), 2, cv2.LINE_AA)
    else:
        pass
    
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='All_gest.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

#Initialize the Flask app
app = Flask(__name__)
camera = cv2.VideoCapture(0)
    
    
def gen_frames():  
    with GestureRecognizer.create_from_options(options) as recognizer:
        # additional_image = cv2.imread('Thumbs Up.jpg')  # Replace 'path_to_additional_image.jpg' with the actual image file path
        # additional_image = cv2.resize(additional_image, (frame.shape[1], frame.shape[0]))
        # additional_image = cv2.resize(additional_image, (frame.shape[1], frame.shape[0]))

        # additional_frame = cv2.resize(additional_frame, (640, 480))

        while True:
            success, frame = camera.read()  # read the camera frame
            if not success:
                break
            else:
                x, y, c = frame.shape
                frame = cv2.flip(frame, 1)
                global className
                if className !='':
                    try:
                        
                        additional_image = cv2.imread(f'data/{className}.jpg')  # Replace 'path_to_additional_image.jpg' with the actual image file path        
                        additional_image = cv2.resize(additional_image, (100, 100))                # additional_image = cv2.resize(additional_image, (640, 480))
                        # frame_with_additional = cv2.addWeighted(frame, 0.7, additional_image, 0.3, 0)
                        x_offset = 10
                        y_offset = 10
                        frame[y_offset:y_offset+additional_image.shape[0], x_offset:x_offset+additional_image.shape[1]] = additional_image
                        
                    except:
                        print(Exception)
                
                # combined_frame = cv2.hconcat([frame, additional_image])
                ret, buffer = cv2.imencode('.jpg', frame)
                fram = buffer.tobytes()
                # yield (b'--frame\r\n'
                #        b'Content-Type: image/jpeg\r\n\r\n' + fram + b'\r\n')  
                
                
                framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(framergb)
                
                if result.multi_hand_landmarks:
                    landmarks = []

                    for handslms in result.multi_hand_landmarks:
                        for lm in handslms.landmark:
                            # print(id, lm)
                            lmx = int(lm.x * x)
                            lmy = int(lm.y * y)

                            landmarks.append([lmx, lmy])
                        

                        frame = np.array(frame, dtype=np.uint8)  # Convert to np.uint8
                        current_timestamp = int(time.time() * 100)  # Convert current time to milliseconds
                        global prev_timestamp
                        if current_timestamp <= prev_timestamp:
                            current_timestamp = prev_timestamp + 1
                        timestamp = current_timestamp
                        prev_timestamp = current_timestamp
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                        recognizer.recognize_async(mp_image, timestamp)
                    
                        

                        # Drawing landmarks on frames
                        # mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                else:
                    className = ''
                
                        # Predict gesture
                yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + fram + b'\r\n')
            # yield (b'--frame\r\n'
            #        b'Content-Type: image/jpeg\r\n\r\n' + additional_image_bytes + b'\r\n')
                   


                        

                # show the prediction on the frame
                # cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                #                 1, (0,0,255), 2, cv2.LINE_AA)

                # Show the final output
                # cv2.imshow("Output", frame) 

                # if cv2.waitKey(1) == ord('q'):
                    # break

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, port=8000)

