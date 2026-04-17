from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO

app = FastAPI()
model = load_model('mnist_model.h5')

# Простая HTML страница
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Распознавание цифр</title>
</head>
<body>
    <h2>Распознавание рукописных цифр</h2>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Распознать</button>
    </form>
    <div id="result"></div>

    <script>
        document.querySelector('form').onsubmit = async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const response = await fetch('/predict', {method: 'POST', body: formData});
            const result = await response.json();
            document.getElementById('result').innerHTML = `<h3>Цифра: ${result.digit}</h3><p>Уверенность: ${(result.confidence * 100).toFixed(1)}%</p>`;
        };
    </script>
</body>
</html>
"""


@app.get("/")
async def root():
    return HTMLResponse(HTML)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Чтение и обработка изображения
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert('L')  # В оттенки серого
    image = image.resize((28, 28))  # Изменение размера
    image_array = 1 - np.array(image) / 255.0  # Нормализация и инверсия
    image_array = image_array.reshape(1, 28, 28, 1)

    # Предсказание
    prediction = model.predict(image_array)[0]
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {"digit": digit, "confidence": confidence}