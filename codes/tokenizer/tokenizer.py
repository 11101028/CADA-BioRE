import jieba
from functools import partial
from transformers import BertTokenizer
from constant import spot_prompt, asoc_prompt, text_start, left_bracket, right_bracket

class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = partial(jieba.cut, HMM=False)

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens

class AddedT5PegasusTokenizer:
    @staticmethod
    def from_pretrained(model_path):
        tokenizer = T5PegasusTokenizer.from_pretrained(model_path)
        tokenizer.add_tokens([spot_prompt, asoc_prompt, text_start, left_bracket, right_bracket],special_tokens=True)
        return tokenizer