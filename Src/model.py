import tensorflow as tf 
import sys
sys.path.append(".")
from Src import config

max_seq_len = config.MAX_SEQ_LEN
Bert_Layer=config.Bert_Layer

def create_model():
    input_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32,
                                       name="input_word_ids")
    attention_mask = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32,
                                    name="input_mask")
    token_type_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32,
                                        name="input_type_ids")
    
    pooled_output, sequence_output = Bert_Layer([input_ids, attention_mask, token_type_ids])

    start_logits = tf.keras.layers.Dense(1, name="start_logit", use_bias=False)(sequence_output)
    start_logits = tf.keras.layers.Flatten()(start_logits)
    end_logits = tf.keras.layers.Dense(1, name="end_logit", use_bias=False)(sequence_output)
    end_logits = tf.keras.layers.Flatten()(end_logits)
    start_probs = tf.keras.layers.Activation(tf.keras.activations.softmax)(start_logits)
    end_probs = tf.keras.layers.Activation(tf.keras.activations.softmax)(end_logits)

    model = tf.keras.Model(inputs={
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids
    },
    outputs=(start_probs, end_probs))
    return model