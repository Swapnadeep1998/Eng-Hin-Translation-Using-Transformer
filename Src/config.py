import tensorflow_hub as hub
import os
from tokenizers import BertWordPieceTokenizer

BERT_HUB_LINK="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
MAX_SEQ_LEN=384
TRAIN_DATA_PATH="Datasets/SQUAD-Train.json"
VAL_DATA_PATH="Datasets/SQUAD-Dev.json"
BATCH_SIZE=16
EPOCHS=4
MODEL_PATH="./Artefacts/model.h5"

Bert_Layer = hub.KerasLayer(BERT_HUB_LINK, trainable=True)
Vocab_File = Bert_Layer.resolved_object.vocab_file.asset_path.numpy().decode("utf-8")
Tokenizer = BertWordPieceTokenizer(Vocab_File, lowercase=True)