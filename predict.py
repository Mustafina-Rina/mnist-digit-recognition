from tensorflow.keras.models import load_model
import numpy as np
from io import BytesIO
from PIL import Image

INPUT_SHAPE = (28, 28)
model = load_model('mnist_model.h5')
def process(image_file):
    image = Image.open(BytesIO(image_file)).convert('L')
    image = image.resize(INPUT_SHAPE)
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    prediction = model.predict(img_array)
    digit = int(np.argmax(prediction))
    return digit