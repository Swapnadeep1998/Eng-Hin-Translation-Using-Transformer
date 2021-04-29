from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from Inference.predictor import Predictor
from Src import config
# import nest_asyncio
# from pyngrok import ngrok

class Input(BaseModel):
    context: str
    question: str

def create_raw_data(context, question):
    raw_data = {"data":
    [
    {"title": "Bert Qestion Answering",
    "paragraphs": [
    {
    "context": context,
    "qas": [
    {"question": question,
    "id": "Q1"
    }
    ]}]}]}
    return raw_data

predictor = Predictor()

app = FastAPI()


@app.post("/")
async def predict_answer(inp:Input):
    context = inp.context
    question = inp.question
    raw_data = create_raw_data(context, question)
    output = Predictor.predict(raw_data)
    return output
    

    
if __name__=="__main__":    
    # ngrok_tunnel = ngrok.connect(8000)
    # print('Public URL:', ngrok_tunnel.public_url)
    # nest_asyncio.apply()
    uvicorn.run("app:app", host='0.0.0.0', port=8000, reload=True)