import cv2
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm

from tensorflow.keras.applications.efficientnet import preprocess_input

from utils.segmentation import segment_leaf

IMG_SIZE = 224

def generate_gradcam(model, img_path):

   

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

    

    last_conv_layer = "top_conv"



    grad_model = tf.keras.models.Model(
        [model.inputs],
        [
            model.get_layer(last_conv_layer).output,
            model.output
        ]
    )

   

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    

    heatmap = np.maximum(heatmap, 0)

    heatmap = heatmap / np.max(heatmap)

    

    heatmap = cv2.resize(
        heatmap,
        (IMG_SIZE, IMG_SIZE)
    )

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

   

    original_img = cv2.imread(img_path)

    original_img = cv2.resize(
        original_img,
        (IMG_SIZE, IMG_SIZE)
    )

   

    superimposed_img = cv2.addWeighted(
        original_img,
       0.75,
heatmap,
0.25,
        0
    )

    return cv2.cvtColor(
        superimposed_img,
        cv2.COLOR_BGR2RGB
    )