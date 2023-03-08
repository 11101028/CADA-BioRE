import torch
import torch.nn as nn
import os
import json
import jsonlines
import shutil
import math
import numpy as np
import torch.nn.functional as F
from d2l import torch as d2l
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import ProgressBar, TokenRematch, get_time, save_args
from metrics import er_metric, re_metric, gen_metric, rc_metric, p2so_metric
from optimizer import GPLinkerOptimizer
from loss import multilabel_categorical_crossentropy, sparse_multilabel_categorical_crossentropy


def kl_div_for_gplinker(a,b,reduction='sum',rdrop_type='softmax'):
    if rdrop_type=='softmax':
        a_2 = a.softmax(dim=-1).reshape(-1)
        b_2 = b.softmax(dim=-1).reshape(-1)
    else:
        a_2 = torch.sigmoid(a).reshape(-1)
        b_2 = torch.sigmoid(b).reshape(-1)
    a = a.reshape(-1)
    b = b.reshape(-1)
    kl_val = torch.dot(a_2-b_2,a-b)
    if reduction != 'sum':
        kl_val = kl_val/a.shape[0]
    return kl_val


class Trainer(object):
    def __init__(
            self,
            args,
            data_processor,
            logger,
            model=None,
            tokenizer=None,
            train_dataset=None,
            eval_dataset=None,
    ):

        self.args = args
        self.model = model
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        
        if train_dataset is not None and isinstance(train_dataset, Dataset):
            self.train_dataset = train_dataset

        if eval_dataset is not None and isinstance(eval_dataset, Dataset):
            self.eval_dataset = eval_dataset

        self.logger = logger

    def train(self):
        args = self.args
        logger = self.logger
        model = self.model
        self.output_dir = os.path.join(args.output_dir, args.time)

        
        if args.distributed == True:
            model = nn.DataParallel(model, device_ids=args.devices).to(args.device)
        else:
            model.to(args.device)
            
        
        
        train_dataloader = self.get_train_dataloader()

        num_training_steps = len(train_dataloader) * args.epochs
        num_warmup_steps = num_training_steps * args.warmup_proportion
        num_examples = len(train_dataloader.dataset)
        
        if args.method_name == 'dual-re' or args.method_name == 'gper':
            optimizer = GPLinkerOptimizer(args,model, train_steps= len(train_dataloader)  * args.epochs)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                        num_training_steps=num_training_steps)
        else:
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.args.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                        num_training_steps=num_training_steps)

        logger.info("***** Running training *****")
        logger.info("Num samples %d", num_examples)
        logger.info("Num epochs %d", args.epochs)
        logger.info("Num training steps %d", num_training_steps)
        logger.info("Num warmup steps %d", num_warmup_steps)

        global_step = 0
        best_step = None
        best_score = 0
        cnt_patience = 0
        
        animator = d2l.Animator(xlabel='epoch', xlim=[0, args.epochs], ylim=[0, 1], fmts=('k-', 'r--', 'y-.', 'm:', 'g--', 'b-.', 'c:'),
                                legend=[f'train loss/{args.loss_show_rate}', 'train_p', 'train_r', 'train_f1', 'val_p', 'val_r', 'val_f1'])

        metric = d2l.Accumulator(5)
        num_batches = len(train_dataloader)
        

        
        for epoch in range(args.epochs):
            pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
            for step, item in enumerate(train_dataloader):
                loss, train_p, train_r, train_f1 = self.training_step(model, item)
                logger.info('loss:{}   p:{}   r:{}   f1:{}'.format(loss,train_p,train_r,train_f1))
                loss = loss.item()
                metric.add(loss, train_p, train_r, train_f1, 1)
                pbar(step, {'loss': loss})

                if args.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                # if args.method_name != 'dual-re' and args.method_name != 'gper':
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.method_name == 'No Valid':
                        animator.add(
                                global_step / num_batches, 
                                (loss / args.loss_show_rate,0, 0, 0, 0, 0, 0))
                        if not os.path.exists(self.output_dir):
                            os.makedirs(self.output_dir)
                        d2l.plt.savefig(os.path.join(self.output_dir, 'training_process.jpg'), dpi=300)
                    else:
                        val_p, val_r, val_f1 = self.evaluate(model)
                        animator.add(
                            global_step / num_batches, 
                            (# metric[0] / metric[-1] / args.loss_show_rate, # loss is too large. Only by dividing loss_show_rate can you see it in the range of [0,1]
                             loss / args.loss_show_rate,
                             train_p,  # metric[1] / metric[-1],
                             train_r,  # metric[2] / metric[-1],
                             train_f1, # metric[3] / metric[-1],
                             val_p,
                             val_r,
                             val_f1))
                        if not os.path.exists(self.output_dir):
                            os.makedirs(self.output_dir)
                        d2l.plt.savefig(os.path.join(self.output_dir, 'training_process.jpg'), dpi=300)

                        if args.save_metric == 'step':
                            save_metric = global_step
                        elif args.save_metric == 'epoch':
                            save_metric = epoch
                        elif args.save_metric == 'loss':
                            save_metric = math.exp(- loss / 10) # math.exp(- metric[0] / metric[-1] / 10)
                        elif args.save_metric == 'p':
                            save_metric = val_p
                        elif args.save_metric == 'r':
                            save_metric = val_r
                        elif args.save_metric == 'f1':
                            save_metric = val_f1

                        if save_metric > best_score:
                            best_score = save_metric
                            best_step = global_step
                            cnt_patience = 0
                            self.args.loss = loss # metric[0] / metric[-1]
                            self.args.train_p, self.args.train_r, self.args.train_f1 = train_p, train_r, train_f1
                                             #  metric[1] / metric[-1], metric[2] / metric[-1], metric[3] / metric[-1]
                            self.args.val_p, self.args.var_r, self.args.val_f1 = val_p, val_r, val_f1
                            self._save_checkpoint(model)
                        else:
                            cnt_patience += 1
                            self.logger.info("Earlystopper counter: %s out of %s", cnt_patience, args.earlystop_patience)
                            if cnt_patience >= self.args.earlystop_patience:
                                break
            if args.method_name == 'No Valid':
                self.args.loss = loss
                self.args.train_p, self.args.train_r, self.args.train_f1 = train_p, train_r, train_f1
                self._save_checkpoint(model)
        
        logger.info(f"\n***** {args.finetuned_model_name} model training stop *****" )
        logger.info(f'finished time: {get_time()}')
        logger.info(f"best val_{args.save_metric}: {best_score}, best step: {best_step}\n" )


        return global_step, best_step

    def evaluate(self, model):
        raise NotImplementedError

    def _save_checkpoint(self, model):
        args = self.args
        
        if args.distributed:
            model=model.module

        model = model.to(torch.device('cpu'))
        torch.save(model.state_dict(), os.path.join(self.output_dir, 'pytorch_model.pt'))
        self.logger.info('Saving models checkpoint to %s', self.output_dir)
        self.tokenizer.save_vocabulary(save_directory=self.output_dir)
        model.encoder.config.to_json_file(os.path.join(self.output_dir, 'config.json'))
        model = model.to(args.device)
        save_args(args, self.output_dir)
    
    
    def load_checkpoint(self):
        args = self.args
        load_dir = os.path.join(args.output_dir, args.model_version)
        self.logger.info(f'load model from {load_dir}')
        checkpoint = torch.load(os.path.join(load_dir, 'pytorch_model.pt'), map_location=torch.device('cpu'))
        if 'module' in list(checkpoint.keys())[0].split('.'):
            self.model = nn.DataParallel(self.model, device_ids=args.devices).to(args.device)
        self.model.load_state_dict(checkpoint)
    
    def training_step(self, model, item):
        raise NotImplementedError

    def get_train_dataloader(self):
        collate_fn = self.train_dataset.collate_fn if hasattr(self.train_dataset, 'collate_fn') else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

    def get_eval_dataloader(self):
        collate_fn = self.eval_dataset.collate_fn if hasattr(self.eval_dataset, 'collate_fn') else None
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

    def get_test_dataloader(self, test_dataset, batch_size=None):
        collate_fn = test_dataset.collate_fn if hasattr(test_dataset, 'collate_fn') else None
        if not batch_size:
            batch_size = self.args.eval_batch_size

        return DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        
class GPERTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPERTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
    
    def evaluate(self, model):
        isPbar=True
        args = self.args
        logger = self.logger
        test_samples = self.data_processor.get_dev_sample()
        num_examples = len(test_samples)
        logger.info("***** Running Evaluation *****")
        logger.info("Num samples %d", num_examples)
        args = self.args
        model = self.model
        device = args.device
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Evaluating')
        predict_data = []
        gold_data = []
        for step, data in enumerate(test_samples):
            for (gold_sub, gold_obj) in data['entity_list']:
                gold_data.append((step,gold_sub,0))
                gold_data.append((step,gold_obj,1))
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
            token2char_span_mapping = self.tokenizer(text, return_offsets_mapping=True, max_length=args.max_length, truncation=True)["offset_mapping"]
            new_span, entities = [], []
            for i in token2char_span_mapping:
                if i[0] == i[1]:
                    new_span.append([])
                else:
                    if i[0] + 1 == i[1]:
                        new_span.append([i[0]])
                    else:
                        new_span.append([i[0], i[-1] - 1])
            threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            score = model(input_ids, attention_mask, token_type_ids)
            outputs = torch.sigmoid(score[0].data).cpu().numpy()
            subjects, objects = set(), set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for l, h, t in zip(*np.where(outputs > args.predict_threshold)):
                if l == 0:
                    subjects.add((h, t))
                else:
                    objects.add((h, t))

            subject_list, object_list = [], []
            for sh, st in subjects:
                subject_list.append(text[new_span[sh][0]:new_span[st][-1] + 1])
            for oh, ot in objects:
                object_list.append(text[new_span[oh][0]:new_span[ot][-1] + 1])
            for sub in subject_list:
                predict_data.append((step,sub,0))
            for obj in object_list:
                predict_data.append((step,obj,1))

        P = len(set(predict_data) & set(gold_data)) / len(set(predict_data))
        R = len(set(predict_data) & set(gold_data)) / len(set(gold_data))
        F = (2 * P * R) / (P + R)
        print('\nEVAL FINISH:P:{}  R:{}  F:{}'.format(P,R,F))
        return P,R,F

    def training_step(self, model, item):
        model.train()
        device = self.args.device
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = kl_div_for_gplinker(logits[::2],logits[1::2],reduction='mean',rdrop_type=self.args.rdrop_type)

            loss = loss + loss_kl / 2 * self.args.rdrop_alpha
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
     
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True):
        args = self.args
        model = self.model
        device = args.device
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
            token2char_span_mapping = self.tokenizer(text, return_offsets_mapping=True, max_length=args.max_length, truncation=True)["offset_mapping"]
            new_span, entities = [], []
            for i in token2char_span_mapping:
                if i[0] == i[1]:
                    new_span.append([])
                else:
                    if i[0] + 1 == i[1]:
                        new_span.append([i[0]])
                    else:
                        new_span.append([i[0], i[-1] - 1])
            threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            score = model(input_ids, attention_mask, token_type_ids)
            outputs = torch.sigmoid(score[0].data).cpu().numpy()
            subjects, objects = set(), set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for l, h, t in zip(*np.where(outputs > args.predict_threshold)):
                if l == 0:
                    subjects.add((h, t))
                else:
                    objects.add((h, t))

            subject_list, object_list = [], []
            for sh, st in subjects:
                subject_list.append(text[new_span[sh][0]:new_span[st][-1] + 1])
            for oh, ot in objects:
                object_list.append(text[new_span[oh][0]:new_span[ot][-1] + 1])
            data['sub_list'] = subject_list
            data['obj_list'] = object_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self):
        args = self.args
        logger = self.logger
        test_samples = self.data_processor.get_test_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir,args.data_dir,args.model_version+'_'+str(args.predict_threshold))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir = os.path.join(output_dir, 'entity_list.jsonl')
        if os.path.exists(output_dir):
            print('already predicted')
            return
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # the preduct result of class 0
            predict_data0 = self._get_predict_entity_list(test_samples)
            for data in predict_data0:
                f.write(data)
        save_args(args,os.path.join(args.result_output_dir,args.data_dir,args.model_version+'_'+str(args.predict_threshold)))
                
    def predict(self):
        self.predict_entity()

    def predict_entity_for_dev(self):
        args = self.args
        logger = self.logger
        test_samples = self.data_processor.get_dev_sample_for_predict()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir,args.data_dir,args.model_version+'_'+str(args.predict_threshold))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir = os.path.join(output_dir, 'entity_list_'+args.dev_name.split('.')[0]+'.jsonl')
        if os.path.exists(output_dir):
            print('already predicted')
            return
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # the preduct result of class 0
            predict_data0 = self._get_predict_entity_list(test_samples)
            for data in predict_data0:
                f.write(data)
        save_args(args,os.path.join(args.result_output_dir,args.data_dir,args.model_version+'_'+str(args.predict_threshold)))
                
    def predict_for_dev(self):
        self.predict_entity_for_dev()

    def predict_entity_for_heatmap(self):
        args = self.args
        logger = self.logger
        test_samples = self.data_processor.get_dev_sample_for_predict()
        num_examples = len(test_samples)
        assert num_examples == 1
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir,args.data_dir,args.model_version+'_'+str(args.predict_threshold))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir = os.path.join(output_dir, 'entity_list_'+args.dev_name.split('.')[0]+'.jsonl')
        if os.path.exists(output_dir):
            print('already predicted')
            return
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # the preduct result of class 0
            predict_data0 = self._get_predict_entity_list(test_samples)
            for data in predict_data0:
                f.write(data)
        save_args(args,os.path.join(args.result_output_dir,args.data_dir,args.model_version+'_'+str(args.predict_threshold)))
                
    def predict_for_heatmap(self):
        self.predict_entity_for_heatmap()
        
class RETrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
    ):
        super(RETrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )

    def training_step(self, model, item):
        model.train()

        
        input_ids, token_type_ids, attention_mask, flag, label = item

        input_ids, token_type_ids, attention_mask, flag, label = input_ids.to(self.args.device), \
                                                                 token_type_ids.to(self.args.device), \
                                                                 attention_mask.to(self.args.device), \
                                                                 flag.to(self.args.device), label.to(self.args.device)
        loss, logits = model(input_ids, token_type_ids, attention_mask, flag, label)
        loss = loss.mean()
        loss.backward()
        
        preds = logits.argmax(axis=1).detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        p, r, f1, _ = re_metric(preds, label)
        return loss.detach(), p, r, f1

    def evaluate(self, model):
        args = self.args
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)

        preds = None
        eval_labels = None

        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()

            input_ids, token_type_ids, attention_mask, flag, label = item

            input_ids, token_type_ids, attention_mask, flag, label = input_ids.to(self.args.device), \
                                                                     token_type_ids.to(self.args.device), \
                                                                     attention_mask.to(self.args.device), \
                                                                     flag.to(self.args.device), label.to(self.args.device)

            with torch.no_grad():
                loss, logits = model(input_ids, token_type_ids, attention_mask, flag, label, mode='dev')

            if preds is None:
                preds = logits.detach().cpu().numpy()
                eval_labels = label.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                eval_labels = np.append(eval_labels, label.detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)
        p, r, f1, _ = re_metric(preds, eval_labels)
        logger.info("%s precision: %s - recall: %s - f1 score: %s", args.finetuned_model_name, p, r, f1)
        return p, r, f1

    def predict(self, test_samples, model, re_dataset_class):
        args = self.args
        logger = self.logger
        model.to(args.device)
        model.eval()
        
        logger.info("***** Running prediction *****")
        output_dir = os.path.join(args.result_output_dir,args.data_dir,args.model_version+'_'+args.prev_model_version+'_'+str(args.predict_threshold))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir = os.path.join(output_dir, 'CMeIE_test.jsonl')
        if os.path.exists(output_dir):
            print('already predicted')
            return
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            pbar = ProgressBar(n_total=len(test_samples), desc='Predicting')
            for step, data in enumerate(test_samples):
                pbar(step)
                results, outputs = self.data_processor.build_text(data)
                spo_list = [re['spo_list'] for re in results]
                temp_re_dataset = re_dataset_class(outputs, data_processor=self.data_processor,
                                                   tokenizer=self.tokenizer, max_length=args.max_length, mode="test")
                logits = []
                probs = []
                with torch.no_grad():
                    for item in temp_re_dataset:
                        input_ids, token_type_ids, attention_mask, flag = item
                        input_ids, token_type_ids, attention_mask, flag = input_ids.to(args.device), \
                                                                          token_type_ids.to(args.device), \
                                                                          attention_mask.to(args.device), \
                                                                          flag.to(args.device)
                        logit = model(input_ids=input_ids.view(1, -1), token_type_ids=token_type_ids.view(1, -1),
                                          attention_mask=attention_mask.view(1, -1),
                                          flag=flag.view(1, -1))  # batch, labels
                        
                        prob = round(nn.functional.softmax(logit).max().item(),5)
                        probs.append(prob)
                        logit = logit.argmax(dim=-1).squeeze(-1)  # batch,
                        logits.append(logit.detach().cpu().item())
                for i in range(len(temp_re_dataset)):
                    if logits[i] > 0:
                        spo_list[i]['predicate'] = self.data_processor.id2predicate[logits[i]]
                        spo_list[i]['prob'] = probs[i]

                new_spo_list = []
                for spo in spo_list:
                    if 'predicate' in spo.keys():
                        combined = True
                        for text in data['text'].split("。"):
                            if spo['object'] in text and spo['subject'] in text:
                                combined = False
                                break
                        tmp = {}
                        tmp['prob'] = spo['prob']
                        tmp['Combined'] = combined
                        tmp['predicate'] = spo['predicate'].split('|')[0]
                        tmp['subject'] = spo['subject']
                        tmp['subject_type'] = self.data_processor.pre_sub_obj[spo['predicate']][0]
                        tmp['object'] = {'@value': spo['object']}
                        tmp['object_type'] = {'@value': self.data_processor.pre_sub_obj[spo['predicate']][1]}
                        new_spo_list.append(tmp)

                new_spo_list2 = []  # delete duplicate data
                for s in new_spo_list:
                    if s not in new_spo_list2:
                        new_spo_list2.append(s)

                for i in range(len(new_spo_list2)):
                    if 'object' not in new_spo_list2[i].keys():
                        del new_spo_list2[i]

                tmp_result = dict()
                tmp_result['text'] = data['text']
                tmp_result['spo_list'] = new_spo_list2
                f.write(tmp_result)
        save_args(args,os.path.join(args.result_output_dir,args.data_dir,args.model_version+'_'+args.prev_model_version+'_'+str(args.predict_threshold)))
    
    def predict_for_dev(self, test_samples, model, re_dataset_class):
        args = self.args
        logger = self.logger
        model.to(args.device)
        model.eval()
        
        logger.info("***** Running prediction *****")
        output_dir = os.path.join(args.result_output_dir,args.data_dir,args.model_version+'_'+args.prev_model_version+'_'+str(args.predict_threshold))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir = os.path.join(output_dir, 'CMeIE_'+args.dev_name.split('.')[0]+'.jsonl')
        if os.path.exists(output_dir):
            print('already predicted')
            return
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            pbar = ProgressBar(n_total=len(test_samples), desc='Predicting')
            for step, data in enumerate(test_samples):
                pbar(step)
                results, outputs = self.data_processor.build_text(data)
                spo_list = [re['spo_list'] for re in results]
                temp_re_dataset = re_dataset_class(outputs, data_processor=self.data_processor,
                                                   tokenizer=self.tokenizer, max_length=args.max_length, mode="test")
                logits = []
                probs = []
                with torch.no_grad():
                    for item in temp_re_dataset:
                        input_ids, token_type_ids, attention_mask, flag = item
                        input_ids, token_type_ids, attention_mask, flag = input_ids.to(args.device), \
                                                                          token_type_ids.to(args.device), \
                                                                          attention_mask.to(args.device), \
                                                                          flag.to(args.device)
                        logit = model(input_ids=input_ids.view(1, -1), token_type_ids=token_type_ids.view(1, -1),
                                          attention_mask=attention_mask.view(1, -1),
                                          flag=flag.view(1, -1))  # batch, labels
                        
                        prob = round(nn.functional.softmax(logit).max().item(),5)
                        probs.append(prob)
                        logit = logit.argmax(dim=-1).squeeze(-1)  # batch,
                        logits.append(logit.detach().cpu().item())
                for i in range(len(temp_re_dataset)):
                    if logits[i] > 0:
                        spo_list[i]['predicate'] = self.data_processor.id2predicate[logits[i]]
                        spo_list[i]['prob'] = probs[i]

                new_spo_list = []
                for spo in spo_list:
                    if 'predicate' in spo.keys():
                        combined = True
                        for text in data['text'].split("。"):
                            if spo['object'] in text and spo['subject'] in text:
                                combined = False
                                break
                        tmp = {}
                        tmp['prob'] = spo['prob']
                        tmp['Combined'] = combined
                        tmp['predicate'] = spo['predicate'].split('|')[0]
                        tmp['subject'] = spo['subject']
                        tmp['subject_type'] = self.data_processor.pre_sub_obj[spo['predicate']][0]
                        tmp['object'] = {'@value': spo['object']}
                        tmp['object_type'] = {'@value': self.data_processor.pre_sub_obj[spo['predicate']][1]}
                        new_spo_list.append(tmp)

                new_spo_list2 = []  # delete duplicate data
                for s in new_spo_list:
                    if s not in new_spo_list2:
                        new_spo_list2.append(s)

                for i in range(len(new_spo_list2)):
                    if 'object' not in new_spo_list2[i].keys():
                        del new_spo_list2[i]

                tmp_result = dict()
                tmp_result['text'] = data['text']
                tmp_result['spo_list'] = new_spo_list2
                f.write(tmp_result)
        save_args(args,os.path.join(args.result_output_dir,args.data_dir,args.model_version+'_'+args.prev_model_version+'_'+str(args.predict_threshold)))
    
    
    def predict_for_heatmap(self, test_samples, model, re_dataset_class):
        args = self.args
        logger = self.logger
        model.to(args.device)
        model.eval()
        
        logger.info("***** Running prediction *****")
        output_dir = os.path.join(args.result_output_dir,args.data_dir,args.model_version+'_'+args.prev_model_version+'_'+str(args.predict_threshold))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir = os.path.join(output_dir, 'CMeIE_'+args.dev_name.split('.')[0]+'.jsonl')
        if os.path.exists(output_dir):
            print('already predicted')
            return
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            pbar = ProgressBar(n_total=len(test_samples), desc='Predicting')
            for step, data in enumerate(test_samples):
                pbar(step)
                results, outputs = self.data_processor.build_text(data)
                spo_list = [re['spo_list'] for re in results]
                temp_re_dataset = re_dataset_class(outputs, data_processor=self.data_processor,
                                                   tokenizer=self.tokenizer, max_length=args.max_length, mode="test")
                logits = []
                probs = []
                with torch.no_grad():
                    for item in temp_re_dataset:
                        input_ids, token_type_ids, attention_mask, flag = item
                        input_ids, token_type_ids, attention_mask, flag = input_ids.to(args.device), \
                                                                          token_type_ids.to(args.device), \
                                                                          attention_mask.to(args.device), \
                                                                          flag.to(args.device)
                        logit = model(input_ids=input_ids.view(1, -1), token_type_ids=token_type_ids.view(1, -1),
                                          attention_mask=attention_mask.view(1, -1),
                                          flag=flag.view(1, -1))  # batch, labels
                        
                        prob = round(nn.functional.softmax(logit).max().item(),5)
                        probs.append(prob)
                        logit = logit.argmax(dim=-1).squeeze(-1)  # batch,
                        logits.append(logit.detach().cpu().item())
                for i in range(len(temp_re_dataset)):
                    if logits[i] > 0:
                        spo_list[i]['predicate'] = self.data_processor.id2predicate[logits[i]]
                        spo_list[i]['prob'] = probs[i]

                new_spo_list = []
                for spo in spo_list:
                    if 'predicate' in spo.keys():
                        combined = True
                        for text in data['text'].split("。"):
                            if spo['object'] in text and spo['subject'] in text:
                                combined = False
                                break
                        tmp = {}
                        tmp['prob'] = spo['prob']
                        tmp['Combined'] = combined
                        tmp['predicate'] = spo['predicate'].split('|')[0]
                        tmp['subject'] = spo['subject']
                        tmp['subject_type'] = self.data_processor.pre_sub_obj[spo['predicate']][0]
                        tmp['object'] = {'@value': spo['object']}
                        tmp['object_type'] = {'@value': self.data_processor.pre_sub_obj[spo['predicate']][1]}
                        new_spo_list.append(tmp)

                new_spo_list2 = []  # delete duplicate data
                for s in new_spo_list:
                    if s not in new_spo_list2:
                        new_spo_list2.append(s)

                for i in range(len(new_spo_list2)):
                    if 'object' not in new_spo_list2[i].keys():
                        del new_spo_list2[i]

                tmp_result = dict()
                tmp_result['text'] = data['text']
                tmp_result['spo_list'] = new_spo_list2
                f.write(tmp_result)
        save_args(args,os.path.join(args.result_output_dir,args.data_dir,args.model_version+'_'+args.prev_model_version+'_'+str(args.predict_threshold)))

class P2SOTrainer(Trainer):
    def __init__(
            self,
            prefix_mode,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
    ):
        super(P2SOTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        self.prefix_mode = prefix_mode
    
    def evaluate(self, model):
        isPbar=True
        args = self.args
        logger = self.logger
        test_samples = self.data_processor.get_dev_sample()
        # print(test_samples[:3])
        num_examples = len(test_samples)
        logger.info("***** Running Evaluation *****")
        logger.info("Num samples %d", num_examples)
        args = self.args
        model = self.model
        device = args.device
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Evaluating')
        predict_data = []
        gold_data = []
        predict_data_2 = []
        gold_data_2 = []

        for step, data in enumerate(test_samples):
            text = data['text']
            for spo in data['spo_list']:
                sub = spo['subject']
                sub_type = spo['subject_type']
                obj = spo['object']['@value']
                obj_type = spo['object_type']['@value']
                p = spo['predicate'] + '|' + obj_type + '|' + sub_type
                gold_data.append((step,p,sub,0))
                gold_data.append((step,p,obj,1))
                gold_data_2.append((step,sub,0))
                gold_data_2.append((step,obj,1))
            if isPbar:
                pbar(step)
            # build p2so data
            samples = self.data_processor.build_data(data, mode='dev')
            new_instance = {'text':text, 'spo_list':[]}
            p2so_dic = defaultdict(set)
            for sample in samples:
                _, p, so_list = sample.values()
                p2so_dic[p] = set(so_list)
            # according to p build prefix and predict 
            # calculate offset mapping，Since prefix is not considered in the follow-up, it is unnecessary to add it
            token2char_span_mapping = self.tokenizer(text, return_offsets_mapping=True, max_length=args.max_length, truncation=True)["offset_mapping"]
            new_span = []
            for i in token2char_span_mapping:
                if i[0] == i[1]:
                    new_span.append([])
                else:
                    if i[0] + 1 == i[1]:
                        new_span.append([i[0]])
                    else:
                        new_span.append([i[0], i[-1] - 1])
            with torch.no_grad():
                for sample in samples:
                    _, p, so_list = sample.values()
                    predicate, obj_type, subject_type = p.split('|')
                    prefix = self.get_prefix(p)
                    # roberta is separated between two sentences by </s><s>, which is different from bert, which is <sep>.
                    prefix_encode_len = len(self.tokenizer(prefix)['input_ids'])-1

                    encoder_txt = self.tokenizer.encode_plus(prefix,text, max_length=args.max_length, truncation=True)
                    input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
                    token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
                    attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
                    score = model(input_ids, token_type_ids,attention_mask)
                    outputs = torch.sigmoid(score[0].data).cpu().numpy()
                    subjects, objects = set(), set()
                    # delete prefix
                    outputs = outputs[:,prefix_encode_len:,prefix_encode_len:]
                    outputs[:, [0, -1]] -= np.inf
                    outputs[:, :, [0, -1]] -= np.inf

                    # for l, h, t in zip(*np.where(outputs > args.predict_threshold)):
                    for l, h, t in zip(*np.where(outputs > 0.5)):
                        if l == 0:
                            subjects.add((h, t))
                        else:
                            objects.add((h, t))

                    sub_list, obj_list = [], []
                    for sh, st in subjects:
                        sub_list.append(text[new_span[sh][0]:new_span[st][-1] + 1])
                    for oh, ot in objects:
                        obj_list.append(text[new_span[oh][0]:new_span[ot][-1] + 1])

                    for sub in sub_list:
                        predict_data.append((step,p,sub,0))
                    for obj in obj_list:
                        predict_data.append((step,p,obj,1))

                    for sub in sub_list:
                        predict_data_2.append((step,sub,0))
                    for obj in obj_list:
                        predict_data_2.append((step,obj,1))
        
        P = len(set(predict_data) & set(gold_data)) / len(set(predict_data))
        R = len(set(predict_data) & set(gold_data)) / len(set(gold_data))
        F = (2 * P * R) / (P + R)
        print('\nEVAL FINISH:P:{}  R:{}  F:{}'.format(P,R,F))

        P_2 = len(set(predict_data_2) & set(gold_data_2)) / len(set(predict_data_2))
        R_2 = len(set(predict_data_2) & set(gold_data_2)) / len(set(gold_data_2))
        F_2 = (2 * P * R) / (P + R)
        print('\nEVAL FINISH_2:P_2:{}  R_2:{}  F_2:{}'.format(P_2,R_2,F_2))
        return P,R,F

    def training_step(self, model, item):
        model.train()
        device = self.args.device
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = kl_div_for_gplinker(logits[::2],logits[1::2],reduction='mean',rdrop_type=self.args.rdrop_type)

            loss = loss + loss_kl / 2 * self.args.rdrop_alpha
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
     
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    

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

    def _get_predict_entity_list(self, test_samples, isPbar=True):
        args = self.args
        model = self.model
        device = args.device
        num_examples = len(test_samples)
        model.to(device)
        model.eval()
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            # build p2so data
            samples = self.data_processor.build_data(data, mode='test')
            new_instance = {'text':text, 'spo_list':[]}
            p2so_dic = defaultdict(set)
            for sample in samples:
                _, p, so_list = sample.values()
                p2so_dic[p] = set(so_list)
            # according to p build prefix and predict 
            # calculate offset mapping，Since prefix is not considered in the follow-up, it is unnecessary to add it
            token2char_span_mapping = self.tokenizer(text, return_offsets_mapping=True, max_length=args.max_length, truncation=True)["offset_mapping"]
            new_span = []
            for i in token2char_span_mapping:
                if i[0] == i[1]:
                    new_span.append([])
                else:
                    if i[0] + 1 == i[1]:
                        new_span.append([i[0]])
                    else:
                        new_span.append([i[0], i[-1] - 1])
            with torch.no_grad():
                for sample in samples:
                    _, p, so_list = sample.values()
                    predicate, obj_type, subject_type = p.split('|')
                    prefix = self.get_prefix(p)
                    # roberta is separated between two sentences by </s><s>, which is different from bert, which is <sep>.
                    prefix_encode_len = len(self.tokenizer(prefix)['input_ids'])-1

                    encoder_txt = self.tokenizer.encode_plus(prefix,text, max_length=args.max_length, truncation=True)
                    input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
                    token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
                    attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
                    score = model(input_ids, token_type_ids,attention_mask)
                    outputs = torch.sigmoid(score[0].data).cpu().numpy()
                    subjects, objects = set(), set()
                    # delete prefix
                    outputs = outputs[:,prefix_encode_len:,prefix_encode_len:]
                    outputs[:, [0, -1]] -= np.inf
                    outputs[:, :, [0, -1]] -= np.inf





                    # for l, h, t in zip(*np.where(outputs > args.predict_threshold)):
                    for l, h, t in zip(*np.where(outputs > 0.5)):
                        if l == 0:
                            subjects.add((h, t))
                        else:
                            objects.add((h, t))

                    sub_list, obj_list = [], []
                    for sh, st in subjects:
                        sub_list.append(text[new_span[sh][0]:new_span[st][-1] + 1])
                    for oh, ot in objects:
                        obj_list.append(text[new_span[oh][0]:new_span[ot][-1] + 1])
                    for sub, obj, prob in so_list:
                        if (sub not in sub_list or obj not in obj_list) and prob < args.dual_threshold:
                            p2so_dic[p] -= set([(sub, obj, prob)])
            # judge and get results
            for index, spo in enumerate(data['spo_list']):
                predicate = spo['predicate']
                sub_type = spo['subject_type']
                obj_type = spo['object_type']['@value']
                sub, obj = spo['subject'], spo['object']['@value']
                prob = spo['prob']
                p = predicate + '|' + obj_type+ '|' + sub_type
                
                # spo that meet the dual verification and rules are added to the prediction results
                if (sub, obj, prob) in p2so_dic[p] and self.data_processor.regular(spo) :
                    del spo['prob']
                    new_instance['spo_list'].append(spo)
            predict_data.append(new_instance)
        return predict_data
    
    def predict_entity(self):
        args = self.args
        logger = self.logger
        test_samples = self.data_processor.get_test_sample(args.ori_path)
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir,args.data_dir,args.model_version+'_'+args.prev_model_version+'_'+args.prev_prev_model_version+'_'+str(args.predict_threshold)+'_'+str(args.dual_threshold)+'_'+args.prefix_mode)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir = os.path.join(output_dir, 'CMeIE_test.jsonl')
        if os.path.exists(output_dir):
            print('already predicted')
            return
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # prediction results for class 0 data
            predict_data0 = self._get_predict_entity_list(test_samples)
            for data in predict_data0:
                f.write(data)
        save_args(args,os.path.join(args.result_output_dir,args.data_dir,args.model_version+'_'+args.prev_model_version+'_'+args.prev_prev_model_version+'_'+str(args.predict_threshold)+'_'+str(args.dual_threshold)+'_'+args.prefix_mode))
        with open(os.path.join(args.result_output_dir,args.data_dir,args.model_version+'_'+args.prev_model_version+'_'+args.prev_prev_model_version+'_'+str(args.predict_threshold)+'_'+str(args.dual_threshold)+'_'+args.prefix_mode,'all_model_versions.txt'),'w') as f:
            f.write('gper_model_version:{}\n'.format(args.prev_prev_model_version))
            f.write('re_model_version:{}\n'.format(args.prev_model_version))
            f.write('p2so_model_version:{}\n'.format(args.model_version))
            f.write('predict_threshold:{}\n'.format(args.predict_threshold))
            f.write('dual_threshold:{}\n'.format(args.dual_threshold))
            f.write('prefix_mode:{}\n'.format(args.prefix_mode))
            if args.do_rdrop:
                f.write('do_rdrop:True')
                f.write('rdrop_type:{}'.format(args.rdrop_type))
            else:
                f.write('do_rdrop:False')
            
        if args.data_dir=='./CMeIE':
            return
        self.get_metrics()
    
                    
    def predict(self):
        self.predict_entity()

    def predict_entity_for_dev(self):
        args = self.args
        logger = self.logger
        test_samples = self.data_processor.get_test_sample(args.ori_path)
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir,args.data_dir,args.model_version+'_'+args.prev_model_version+'_'+args.prev_prev_model_version+'_'+str(args.predict_threshold)+'_'+str(args.dual_threshold)+'_'+args.prefix_mode)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir = os.path.join(output_dir, 'CMeIE_'+args.dev_name.split('.')[0]+'.jsonl')
        if os.path.exists(output_dir):
            print('already predicted')
            return
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # Prediction results for class 0 data
            predict_data0 = self._get_predict_entity_list(test_samples)
            for data in predict_data0:
                f.write(data)
        save_args(args,os.path.join(args.result_output_dir,args.data_dir,args.model_version+'_'+args.prev_model_version+'_'+args.prev_prev_model_version+'_'+str(args.predict_threshold)+'_'+str(args.dual_threshold)+'_'+args.prefix_mode))
        with open(os.path.join(args.result_output_dir,args.data_dir,args.model_version+'_'+args.prev_model_version+'_'+args.prev_prev_model_version+'_'+str(args.predict_threshold)+'_'+str(args.dual_threshold)+'_'+args.prefix_mode,'all_model_versions.txt'),'w') as f:
            f.write('gper_model_version:{}\n'.format(args.prev_prev_model_version))
            f.write('re_model_version:{}\n'.format(args.prev_model_version))
            f.write('p2so_model_version:{}\n'.format(args.model_version))
            f.write('predict_threshold:{}\n'.format(args.predict_threshold))
            f.write('dual_threshold:{}\n'.format(args.dual_threshold))
            f.write('prefix_mode:{}\n'.format(args.prefix_mode))
            if args.do_rdrop:
                f.write('do_rdrop:True')
                f.write('rdrop_type:{}'.format(args.rdrop_type))
            else:
                f.write('do_rdrop:False')
        self.get_metrics()
    
                    
    def predict_for_dev(self):
        self.predict_entity_for_dev()

    def predict_entity_for_heatmap(self):
        args = self.args
        logger = self.logger
        test_samples = self.data_processor.get_test_sample(args.ori_path)
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir,args.data_dir,args.model_version+'_'+args.prev_model_version+'_'+args.prev_prev_model_version+'_'+str(args.predict_threshold)+'_'+str(args.dual_threshold)+'_'+args.prefix_mode)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir = os.path.join(output_dir, 'CMeIE_'+args.dev_name.split('.')[0]+'.jsonl')
        if os.path.exists(output_dir):
            print('already predicted')
            return
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # Prediction results for class 0 data
            predict_data0 = self._get_predict_entity_list(test_samples)
            for data in predict_data0:
                f.write(data)
        save_args(args,os.path.join(args.result_output_dir,args.data_dir,args.model_version+'_'+args.prev_model_version+'_'+args.prev_prev_model_version+'_'+str(args.predict_threshold)+'_'+str(args.dual_threshold)+'_'+args.prefix_mode))
        with open(os.path.join(args.result_output_dir,args.data_dir,args.model_version+'_'+args.prev_model_version+'_'+args.prev_prev_model_version+'_'+str(args.predict_threshold)+'_'+str(args.dual_threshold)+'_'+args.prefix_mode,'all_model_versions.txt'),'w') as f:
            f.write('gper_model_version:{}\n'.format(args.prev_prev_model_version))
            f.write('re_model_version:{}\n'.format(args.prev_model_version))
            f.write('p2so_model_version:{}\n'.format(args.model_version))
            f.write('predict_threshold:{}\n'.format(args.predict_threshold))
            f.write('dual_threshold:{}\n'.format(args.dual_threshold))
            f.write('prefix_mode:{}\n'.format(args.prefix_mode))
            if args.do_rdrop:
                f.write('do_rdrop:True')
                f.write('rdrop_type:{}'.format(args.rdrop_type))
            else:
                f.write('do_rdrop:False')
        self.get_metrics()
    
                    
    def predict_for_heatmap(self):
        self.predict_entity_for_heatmap()

    def get_metrics(self):
        args = self.args
        gold_path = args.gold_path
        ori_path = args.ori_path
        out_path = args.out_path
        print('gold_path:{}'.format(gold_path))
        print('ori_path:{}'.format(ori_path))
        print('out_path:{}'.format(out_path))
        # obtain gold data
        all_gold_jsons=[]
        with open(gold_path, 'r') as f_1:
            lines = f_1.readlines()
            for line in lines:
                all_gold_jsons.append(json.loads(line))
        gold_spos=[]
        for i in range(len(all_gold_jsons)):
            gold_json=all_gold_jsons[i]
            spo_list=gold_json['spo_list']
            for spo in spo_list:
                gold_spos.append((i,spo['predicate'],spo['subject'],spo['object']['@value']))

        # get the initial forecast data
        all_ori_jsons=[]
        with open(ori_path, 'r') as f_2:
            lines = f_2.readlines()
            for line in lines:
                all_ori_jsons.append(json.loads(line))
        ori_spos=[]
        for i in range(len(all_ori_jsons)):
            predict_json=all_ori_jsons[i]
            spo_list=predict_json['spo_list']
            for spo in spo_list:
                ori_spos.append((i,spo['predicate'],spo['subject'],spo['object']['@value']))
        P = len(set(ori_spos) & set(gold_spos)) / len(set(ori_spos))
        R = len(set(ori_spos) & set(gold_spos)) / len(set(gold_spos))
        F = (2 * P * R) / (P + R)
        print('Initial prediction result\n')
        print('PRE:{}\tREC:{}\tF1:{}'.format(P,R,F))

        # Obtain forecast data after P2SO
        all_predict_jsons=[]
        with open(out_path, 'r') as f_2:
            lines = f_2.readlines()
            for line in lines:
                all_predict_jsons.append(json.loads(line))
        predict_spos=[]
        for i in range(len(all_predict_jsons)):
            predict_json=all_predict_jsons[i]
            spo_list=predict_json['spo_list']
            for spo in spo_list:
                predict_spos.append((i,spo['predicate'],spo['subject'],spo['object']['@value']))

        # calculate pre,rec,f1
        P = len(set(predict_spos) & set(gold_spos)) / len(set(predict_spos))
        R = len(set(predict_spos) & set(gold_spos)) / len(set(gold_spos))
        F = (2 * P * R) / (P + R)
        print('the result of after initial P2SO\n')
        print('PRE:{}\tREC:{}\tF1:{}'.format(P,R,F))




