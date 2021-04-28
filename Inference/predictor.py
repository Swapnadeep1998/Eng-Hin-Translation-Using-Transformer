from Src.config import MODEL_PATH
import tensorflow as tf 
import sys
sys.path.append(".")
from Src.model import create_model
from Src import config
from Src.Input.data import create_squad_inference
import numpy as np

model_path = config.MODEL_PATH

class Predictor:
    def __init__(self):
        self.model_path = model_path
        self.model = create_model()
        self.model.load_weights(model_path)

    def predict(self, raw_data):
        input_data, context_token_to_char, context, question = create_squad_inference(raw_data)
        pred_start, pred_end = self.model.predict(input_data)
        for (start, end) in zip(pred_start, pred_end):    
            offsets = context_token_to_char
            start = np.argmax(start)
            end = np.argmax(end)
            pred_ans = None
            if start >= len(offsets):
                continue
            pred_char_start = offsets[start][0]
            if end < len(offsets):
                pred_ans = context[pred_char_start:offsets[end][1]]
            else:
                pred_ans = context[pred_char_start:]

        return {
            "Question": question,
            "Answer": pred_ans
        }


