from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import FPS
import numpy as np
import imutils
import time
import dlib
import json
import cv2
import sys
from flask import Response
from flask import Flask
from flask import render_template
import threading

from publisher import LocalClient

lock = threading.Lock()

outputFrame = None
configurations = None
client = None
video_stream = None

app = Flask(__name__)


def read_configuration(path):
    global configurations
    print(path)
    with open(path) as json_data_file:
        data = json.load(json_data_file)
        print('[MAIN] reading configuration ok')
        configurations = data


def detect():
    global video_stream, outputFrame, lock, client, configurations

    config = configurations['people_counter']

    classes = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(config["prototxt"], config["model"])

    print("[INFO] opening video file...")
    video_stream = cv2.VideoCapture(config["input"])

    W = None
    H = None
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackable_objects = {}

    total_frames = 0
    total_down = 0
    total_up = 0
    total = 0
    new_person = False
    fps = FPS().start()

    while True:

        frame = video_stream.read()
        frame = frame[1] if "input" in config else frame
        if "input" in config and frame is None:
            break
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        status = "Waiting"
        rects = []

        if total_frames % config["skip_frames"] == 0:
            status = "Detecting"
            trackers = []
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > config["confidence"]:
                    idx = int(detections[0, 0, i, 1])
                    if classes[idx] != "person":
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (start_x, start_y, end_x, end_y) = box.astype("int")

                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(start_x, start_y, end_x, end_y)
                    tracker.start_track(rgb, rect)

                    trackers.append(tracker)
        else:
            for tracker in trackers:
                status = "Tracking"
                tracker.update(rgb)
                pos = tracker.get_position()
                start_x = int(pos.left())
                start_y = int(pos.top())
                end_x = int(pos.right())
                end_y = int(pos.bottom())
                rects.append((start_x, start_y, end_x, end_y))

        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():

            to = trackable_objects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)

            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                if not to.counted:
                    if direction < 0 and centroid[1] < H // 2:
                        total_up += 1
                        total += 1
                        to.counted = True
                        new_person = True

                    elif direction > 0 and centroid[1] > H // 2:
                        total_down += 1
                        total += 1
                        to.counted = True
                        new_person = True

            trackable_objects[objectID] = to

            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        info = [
            ("Up", total_up),
            ("Down", total_down),
            ("Status", status),
        ]

        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if new_person:
            message = {
                "data": {
                    "value": total
                }
            }
            client.publish('people', json.dumps(message))
            new_person = False

            with lock:
                outputFrame = frame.copy()

        total_frames += 1
        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    video_stream.release()


def generate():
    global outputFrame, lock

    while True:
        with lock:

            if outputFrame is None:
                continue

            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            if not flag:
                continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


def infinite_loop():
    while True:
        detect()
        time.sleep(2)
        print("[INFO] replay")


def mqtt_connection():
    global client
    client = LocalClient(configurations['mqtt']['client_id'],
                         configurations['mqtt']['host'],
                         configurations['mqtt']['port'])

    client.run()


if __name__ == '__main__':
    config_file = None
    if len(sys.argv) < 2:
        print("[MAIN] Usage: python3 main.py [config-file-path.json]")
        exit(1)
    else:
        config_file = sys.argv[1]

    read_configuration(config_file)

    t = threading.Thread(target=infinite_loop)
    t.daemon = True
    t.start()

    t1 = threading.Thread(target=mqtt_connection)
    t1.daemon = True
    t1.start()

    app.run(host=configurations["streaming"]["host"], port=configurations["streaming"]["port"], debug=True,
            threaded=True, use_reloader=False)
