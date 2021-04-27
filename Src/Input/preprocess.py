import sys
sys.path.append(".")
from Src import config

max_seq_len=config.MAX_SEQ_LEN
tokenizer=config.Tokenizer

class Sample:
    def __init__(self, question, context,
                 start_char_idx=None, answer_text=None, 
                 all_answers=None):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.skip = False
        self.start_tok_idx = -1
        self.end_tok_idx = -1

        self.inputs = {}
        self.targets = []

    def preprocess(self):
        context = " ".join(str(self.context).split())
        question = " ".join(str(self.question).split())

        tokenized_context = tokenizer.encode(context)
        tokenized_question = tokenizer.encode(question)

        if self.answer_text is not None:
            answer = " ".join(str(self.answer_text).split())
            end_char_idx = self.start_char_idx + len(answer)

            if end_char_idx > len(context):         
                self.skip = True
                return

            is_char_in_index = [0]*len(context)
            for idx in range(self.start_char_idx, end_char_idx):
                is_char_in_index[idx] = 1

            ans_token_idx = []
            for idx,(start, end) in enumerate(tokenized_context.offsets):
                if sum(is_char_in_index[start:end])>0:
                    ans_token_idx.append(idx)
            if len(ans_token_idx)==0:
                self.skip=True
                return
            
            self.start_tok_idx = ans_token_idx[0]
            self.end_tok_idx = ans_token_idx[-1]

        input_ids = tokenized_context.ids + tokenized_question.ids[1:]  # [CLS] context [SEP] Question [SEP]
        token_type_ids = [0]*len(tokenized_context.ids) + [1]*len(tokenized_question.ids[1:])
        attention_mask = [1]*len(input_ids)
        padding_length = max_seq_len - len(input_ids)

        if padding_length > 0:
            input_ids = input_ids + ([0]*padding_length)
            token_type_ids = token_type_ids + ([0]*padding_length)
            attention_mask = attention_mask + ([0]*padding_length)
        else:
            self.skip = True
            return 
        
        self.inputs["input_ids"]=input_ids
        self.inputs["token_type_ids"] = token_type_ids
        self.inputs["attention_mask"] = attention_mask

        self.targets = (self.start_tok_idx, self.end_tok_idx)              