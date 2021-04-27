import sys
sys.path.append(".")
from Src.Input.preprocess import Sample
import json
import tensorflow as tf

def create_squad_examples(raw_data):

    def generator():        
        for item in raw_data["data"]:
            for para in item["paragraphs"]:
                context = para["context"]
                for qa in para["qas"]:
                    question = qa["question"]
                    if "answers" in qa:
                        answer_text = qa["answers"][0]["text"]
                        all_answers = [_["text"] for _ in qa["answers"]]
                        start_char_idx = qa["answers"][0]["answer_start"]
                        squad_eg = Sample(question, context, start_char_idx, answer_text, all_answers)
                    else:
                        squad_eg = Sample(question, context)
                    squad_eg.preprocess()
                    if squad_eg.skip == False:
                        yield (squad_eg.inputs, squad_eg.targets)
                    
    return generator


class DataLoader:
    def __init__(self,data_path):
        with open(data_path, "r") as f:
            self.raw_data = json.load(f)
        self.datagen = create_squad_examples(self.raw_data)

    def get_data(self, shuffle=False, batch_size=32, buffer_size=2000):
        dataset = tf.data.Dataset.from_generator(self.datagen, output_types=(
            {"input_ids": tf.int32, "token_type_ids": tf.int32, "attention_mask": tf.int32},
            (tf.int32, tf.int32)))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size,drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset