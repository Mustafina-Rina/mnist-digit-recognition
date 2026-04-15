from fastapi import FastAPI, File, UploadFile
from predict import process
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    result = process(contents)
    return {"digit": result}