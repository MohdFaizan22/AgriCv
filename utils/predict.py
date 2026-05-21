import numpy as np
import cv2

from tensorflow.keras.applications.efficientnet import preprocess_input

from utils.segmentation import segment_leaf

IMG_SIZE = 224

def predict_disease(model, img_path, class_names):


    segmented_img = segment_leaf(img_path)

    segmented_img = cv2.resize(
        segmented_img,
        (IMG_SIZE, IMG_SIZE)
    )

   

    img_array = np.array(segmented_img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(img_array)

   
    predictions = model.predict(img_array)[0]


    top3_idx = predictions.argsort()[-3:][::-1]

    results = []

    for idx in top3_idx:

        results.append({

            "disease": class_names[idx],

            "confidence": round(
                float(predictions[idx] * 100),
                2
            )

        })

    return results