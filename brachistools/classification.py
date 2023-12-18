from keras.models import load_model
import numpy as np
import cv2

global classification_model


def load_classification_model(input_image):
    model = load_model('brachistools/models/model.h5')
    classes = ['benign', 'malignant']
    predict_results = model.predict(np.expand_dims(input_image, axis=0), verbose=0)
    predict_class = classes[np.argmax(predict_results)]
    confidence_score = format(np.max(predict_results), '.4f')
    return predict_class, confidence_score


def classification_pipeline(input_image):
    global classification_model
    try:
        classification_model
    except NameError:
        classification_model = load_classification_model()

    input_image = cv2.resize(input_image, (224, 224))
    img = np.array(input_image)

    return classification_model(img)