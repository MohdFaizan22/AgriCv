from rembg import remove
from PIL import Image
import numpy as np

def segment_leaf(image_path):

    input_image = Image.open(image_path)

    output = remove(input_image)

    output = output.convert("RGB")

    segmented_array = np.array(output)

    return segmented_array