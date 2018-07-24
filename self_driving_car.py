import numpy as np
import cv2
import scipy as sp
from keras.models import load_model

model = load_model('model_adam_mse.h5')


def process_image(img):
    image_x = 100
    image_y = 100
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    return np.reshape(img, (-1, image_x, image_y, 1))

def predict(model, image):
    processed = process_image(image)
    steering_angle = float(model.predict(processed, batch_size=1))
    return steering_angle * 60 + sp.pi


if __name__ == "__main__":
    steer = cv2.imread('steering_wheel.jpg', 0)
    rows, cols = steer.shape
    smoothed_angle = 0

    cap = cv2.VideoCapture('run.mp4')
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            gray = cv2.resize((cv2.cvtColor(frame, cv2.COLOR_RGB2HSV))[:, :, 1], (100, 100))
            steering_angle = predict(model, gray)
            print("steering angle:", steering_angle)
            cv2.imshow('frame', cv2.resize(frame, (600, 400), interpolation=cv2.INTER_AREA))
            smoothed_angle += 0.2 * pow(abs((steering_angle - smoothed_angle)), 2.0 / 3.0) * (
                steering_angle - smoothed_angle) / abs(steering_angle - smoothed_angle)
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
            img = cv2.warpAffine(steer, rotation_matrix, (cols, rows))
            cv2.imshow("steering wheel", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
