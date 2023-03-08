import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AdamW
from tqdm import tqdm
import os
import torch.nn.functional as F
from utils import get_devices
from d2l import torch as d2l
from collections import defaultdict

class REModel(nn.Module):
    def __init__(self, tokenizer, encoder_class, num_labels, args):
        super(REModel, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        self.encoder.resize_token_embeddings(len(tokenizer))
        self.classifier = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(in_features=self.encoder.config.hidden_size*2, out_features=num_labels))
        self.args = args

    def forward(self, input_ids, token_type_ids, attention_mask, flag, labels=None, mode='train'):
        device = input_ids.device

        
        outputs = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        last_hidden_state = outputs[0]   # batch, seq, hidden
        batch_size, seq_len, hidden_size = last_hidden_state.shape
        entity_hidden_state = torch.Tensor(batch_size, 2*hidden_size) # batch, 2*hidden
        # flag: batch, 2
        for i in range(batch_size):
            sub_start_idx, obj_start_idx = flag[i, 0], flag[i, 1]
            start_entity = last_hidden_state[i, sub_start_idx, :].view(hidden_size, )   # s_start: hidden,
            end_entity = last_hidden_state[i, obj_start_idx, :].view(hidden_size, )   # o_start: hidden,
            entity_hidden_state[i] = torch.cat([start_entity, end_entity], dim=-1)
        entity_hidden_state = entity_hidden_state.to(device)
        logits = self.classifier(entity_hidden_state)
        if labels is not None:
            cal_loss = self.cal_rdrop_loss if self.args.do_rdrop and mode=='train' else self.cal_loss
            return cal_loss(logits, labels), logits
        return logits
    
    def cal_loss(self, logits, labels):
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(logits, labels.view(-1))
    
    def cal_rdrop_loss(self, logits, labels):
        loss_ce = self.cal_loss(logits, labels)
        loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='mean') + \
                  F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='mean')
        return loss_ce + loss_kl / 2 * self.args.rdrop_alpha
    
  
class P2SOModel(nn.Module):
    def __init__(self, encoder_class, args):
        super(P2SOModel, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 2
        self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]

        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        return mention_outputs


class GPERModel(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPERModel, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 2
        self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]

        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        return mention_outputs
    
class RawGlobalPointer(nn.Module):
    def __init__(self, hiddensize, ent_type_size, inner_dim, RoPE=True, tril_mask=True, do_rdrop=False, dropout=0):
        '''
        :param encoder: BERT
        :param ent_type_size:
        :param inner_dim: 64
        '''
        super().__init__()
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = hiddensize
        if do_rdrop:
            self.dense = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2))
        else:
            self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

        self.RoPE = RoPE
        self.trail_mask = tril_mask

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, last_hidden_state,  attention_mask):
        self.device = attention_mask.device
#         last_hidden_state = context_outputs[0]
        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]
        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]
        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12
        # Eliminate the lower triangle
        if self.trail_mask:
            mask = torch.tril(torch.ones_like(logits), -1)
            logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5

