import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class GPERDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128, 
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True,padding='max_length', max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, obj in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            sh = self.data_processor.search(input_ids,sub)
            obj = self.tokenizer.encode(obj, add_special_tokens=False)
            oh = self.data_processor.search(input_ids,obj )
            if sh != -1 and oh != -1:
                spoes.add((sh, sh+len(sub)-1, oh, oh+len(obj)-1))
        entity_labels = [set() for i in range(2)]
        for sh, st, oh, ot in spoes:
            entity_labels[0].add((sh, st)) 
            entity_labels[1].add((oh, ot)) 
        for label in entity_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        return entity_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_labels = []
        for item in examples:
            head_labels, input_ids, attention_mask, token_type_ids = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels
        
class REDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128,
    ):
        super(REDataset, self).__init__()

        self.texts = samples['text']
        self.flags = samples['flag']

        if mode != "test":
            self.labels = samples['label']

        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

    def __getitem__(self, idx):
        text, flag = self.texts[idx], self.flags[idx]
        
        inputs = self.tokenizer.encode_plus(text, max_length=self.max_length, padding='max_length', truncation=True)

        s_encode = self.tokenizer.encode(flag[0])
        s_start_idx = self.data_processor.search(inputs['input_ids'], s_encode[1:-1])

        o_encode = self.tokenizer.encode(flag[1])
        o_start_idx = self.data_processor.search(inputs['input_ids'], o_encode[1:-1])
        if self.mode != "test":
            label = self.labels[idx]
            
            return torch.tensor(inputs['input_ids']), \
                   torch.tensor(inputs['token_type_ids']), \
                   torch.tensor(inputs['attention_mask']), \
                   torch.tensor([s_start_idx, o_start_idx]).long(), \
                   torch.tensor(label).long()
        else:
            return torch.tensor(inputs['input_ids']), \
                   torch.tensor(inputs['token_type_ids']).long(), \
                   torch.tensor(inputs['attention_mask']).float(), \
                   torch.tensor([s_start_idx, o_start_idx]).long()

    def __len__(self):
        return len(self.texts)
  
class P2SODataset(Dataset):
    def __init__(
            self,
            samples,
            prefix_mode,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128, 
    ):
        self.prefix_mode = prefix_mode
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
        
    def __len__(self):
        return len(self.samples)

    def get_prefix(self, p):
        predicate, obj_type, sub_type = p.split('|')
        if self.prefix_mode == 'pm1':
            return f'[unused1]{predicate}[unused2]{obj_type}'
        if self.prefix_mode == 'pm2':
            return f'[unused1]{predicate}[unused2]{sub_type}'
        if self.prefix_mode == 'pm3':
            return f'[unused1]{predicate}[unused2]{sub_type}[unused3]{obj_type}'
        if self.prefix_mode == 'pm4':
            return f'[unused1]{predicate}'
        if self.prefix_mode == 'pm5':
            return f'[unused1]{sub_type}[unused2]{obj_type}'
        return ''
    
    def encoder(self, item):
        text = item["text"]
        p = item["p"]
        prefix = self.get_prefix(p)
        encoder_text = self.tokenizer(prefix,text, return_offsets_mapping=True,padding='max_length', max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, obj in item["so_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            sh = self.data_processor.search(input_ids,sub )
            obj = self.tokenizer.encode(obj, add_special_tokens=False)
            oh = self.data_processor.search(input_ids,obj )
            if sh != -1 and oh != -1:
                spoes.add((sh, sh+len(sub)-1, oh, oh+len(obj)-1))
        entity_labels = [set() for i in range(2)]
        for sh, st, oh, ot in spoes:
            entity_labels[0].add((sh, st)) 
            entity_labels[1].add((oh, ot)) 
        for label in entity_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        return entity_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_labels = []
        for item in examples:
            head_labels, input_ids, attention_mask, token_type_ids = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels

def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)
