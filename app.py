import gradio as gr
import tensorflow as tf

from utils.predict import predict_disease
from utils.gradcam import generate_gradcam
from utils.disease_info import disease_details



model = tf.keras.models.load_model(
    "final_agrivision_model.keras"
)



class_names = list(disease_details.keys())



def predict(img):

    predictions = predict_disease(
        model,
        img,
        class_names
    )

    top_prediction = predictions[0]

    disease_name = top_prediction["disease"]

    confidence = top_prediction["confidence"]

    

    gradcam_image = generate_gradcam(
        model,
        img
    )

    

    info = disease_details[disease_name]

    

    prediction_text = "# Top Predictions\n\n"

    for pred in predictions:

        conf = pred["confidence"]

        bars = "🟩" * int(conf // 10)

        prediction_text += (
            f"### {pred['disease']}\n"
            f"{bars} {conf}%\n\n"
        )

    

    disease_output = f"""
# Disease: {disease_name}

## Description
{info['description']}

## Symptoms
{info['symptoms']}

## Prevention
{info['prevention']}

## Cure
{info['cure']}
"""

    return (
        prediction_text,
        disease_output,
        gradcam_image
    )



with gr.Blocks(
    title="AgriVision AI"
) as demo:

    gr.Markdown(
        """
# 🌿 AgriVision AI

## Plant Disease Detection using Deep Learning

Upload a leaf image to detect plant disease,
view confidence scores, and visualize Grad-CAM.
"""
    )

    with gr.Row():

        image_input = gr.Image(
            type="filepath",
            label="Upload Leaf Image"
        )

    with gr.Row():

        prediction_output = gr.Markdown(
            label="Predictions"
        )

        disease_output = gr.Markdown(
            label="Disease Details"
        )

    gradcam_output = gr.Image(
        label="Grad-CAM Visualization"
    )

    predict_btn = gr.Button(
        "Detect Disease"
    )

    predict_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[
            prediction_output,
            disease_output,
            gradcam_output
        ]
    )



demo.launch(
    share=True
)