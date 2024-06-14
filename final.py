import cv2
import tensorflow as tf
import numpy as np
import time
import os
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWebEngineWidgets import *

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)

DATADIR = 'dataset'
CATEGORIES = os.listdir(DATADIR)

sample_frames = 50  # Set the number of frames to capture
frame_counter = 0
image_samples = []

while frame_counter < sample_frames:
    ret, img = cam.read()
    img = cv2.flip(img, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w + 50, y + h + 50), (255, 0, 0), 2)
        im = gray[y:y + h, x:x + w]

    cv2.imshow('image', img)

    if 'im' in locals() and frame_counter < sample_frames:
        im_array = cv2.resize(im, (50, 50))
        im_array = cv2.cvtColor(im_array, cv2.COLOR_GRAY2RGB)  # Convert to RGB
        im_array = np.expand_dims(im_array, axis=0)  # Add batch dimension
        image_samples.append(im_array)
        frame_counter += 1

    cv2.waitKey(1)

cam.release()
cv2.destroyAllWindows()

# Convert the list of image samples to a numpy array
image_samples = np.concatenate(image_samples, axis=0)

# Load your model
model = tf.keras.models.load_model("CNN.model")

# Make prediction on the entire sample
predictions = model.predict(image_samples)
prediction = list(predictions[0])
print(prediction)
t = prediction.index(max(prediction))
print(CATEGORIES[prediction.index(max(prediction))])

# Define website blocking functionality
sites_to_block = [
    "www.facebook.com",
    "https://www.facebook.com",
    "facebook.com",
    "https://www.facebook.com/",
    
]
Window_host = r"C:\Windows\System32\drivers\etc\hosts"
default_hoster = Window_host
redirect = "127.0.0.1"

# Function to block websites
def block_websites():
    with open(default_hoster, "r+") as hostfile:
        hosts = hostfile.readlines()
        hostfile.seek(0)
        for host in hosts:
            if not any(site in host for site in sites_to_block):
                hostfile.write(host)
        hostfile.truncate()
    for site in sites_to_block:
        with open(default_hoster, "a") as hostfile:
            hostfile.write(redirect + " " + site + "\n")

# Function to unblock websites
def unblock_websites():
    with open(default_hoster, "r+") as hostfile:
        hosts = hostfile.readlines()
        hostfile.seek(0)
        for host in hosts:
            if any(site in host for site in sites_to_block):
                continue
            hostfile.write(host)
        hostfile.truncate()

# Use age prediction to decide whether to block websites
if t <= 3:
    block_websites()
else:
    unblock_websites()
    

# GUI part
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl('http://google.com'))
        self.setCentralWidget(self.browser)
        self.showMaximized()

        # navbar
        navbar = QToolBar()
        self.addToolBar(navbar)

        back_btn = QAction('Back', self)
        back_btn.triggered.connect(self.browser.back)
        navbar.addAction(back_btn)

        forward_btn = QAction('Forward', self)
        forward_btn.triggered.connect(self.browser.forward)
        navbar.addAction(forward_btn)

        reload_btn = QAction('Reload', self)
        reload_btn.triggered.connect(self.browser.reload)
        navbar.addAction(reload_btn)

        home_btn = QAction('Home', self)
        home_btn.triggered.connect(self.navigate_home)
        navbar.addAction(home_btn)

        self.url_bar = QLineEdit()
        self.url_bar.returnPressed.connect(self.navigate_to_url)
        navbar.addWidget(self.url_bar)

        self.browser.urlChanged.connect(self.update_url)

    def navigate_home(self):
        self.browser.setUrl(QUrl('http://google.com'))

    def navigate_to_url(self):
        url = self.url_bar.text()
        self.browser.setUrl(QUrl(url))

    def update_url(self, q):
        self.url_bar.setText(q.toString())

app = QApplication(sys.argv)
QApplication.setApplicationName('Safe Browser')
window = MainWindow()
app.exec_()
