import os
import json
import jsonlines
from collections import defaultdict
import random

class GPERDataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root,'CMeIE_train.json')
        self.dev_path = os.path.join(root, 'CMeIE_dev.json')
        self.test_path = os.path.join(root, 'CMeIE_test.json')
        self.gold_path = os.path.join(root, 'CMeIE_test.json')
        if args.dev_name:
            self.dev_for_predict = os.path.join(root, args.dev_name)
        self.schema_path = os.path.join(root, '53_schemas.json')
        
        
    def get_train_sample(self):
        return self._pre_process(self.train_path, mode='train')

    def get_gold_sample(self):
        return self._pre_process(self.gold_path, mode='train')

    def get_dev_sample(self):
        return self._pre_process(self.dev_path, mode='dev')

    def get_dev_sample_for_predict(self):
        with open(self.dev_for_predict) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list

    def get_test_sample(self):
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list

    def merge(self,data_1,data_2):
        data_1.extend(data_2)
        return data_1

    
    def search(self, sequence, pattern):
        """
        Find the substring pattern from the sequence
        If found, returns the first index; Otherwise return 0.
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
        
    def _pre_process(self, path, mode):
        new_data = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if mode == 'train':
                random.shuffle(lines)
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in lines:
                num += 1
                line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, 
                        "text": text,
                        "entity_list":[(spo["subject"], spo["object"]["@value"]) for spo in spo_list]
                    })
        return new_data

class REDataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.train_path = os.path.join(root, 'CMeIE_train.json')
        self.dev_path = os.path.join(root, 'CMeIE_dev.json')
        self.test_path = os.path.join(root, 'CMeIE_test.json')

        self.schema_path = os.path.join(root, '53_schemas.json')
        self.pre_sub_obj = None
        self.predicate2id = None
        self.id2predicate = None
        self.s_entity_type = None
        self.o_entity_type = None
        self.args = args
        self._load_schema()

        self.num_labels = len(self.predicate2id.keys())

    def get_train_sample(self):
        return self._pre_process(self.train_path, mode='train')

    def get_dev_sample(self):
        return self._pre_process(self.dev_path, mode='dev')

    def get_test_sample(self, path):
        """ Need new test file generated from the result of ER prediction
        """
        with open(path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            samples = []
            for line in lines:
                data = json.loads(line)
                samples.append(data)
        return samples

    def merge(self,data_1,data_2):
        data_1['text'].extend(data_2['text'])
        data_1['label'].extend(data_2['label'])
        data_1['flag'].extend(data_2['flag'])
        return data_1

    def _pre_process(self, path, mode):
        with open(path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            if mode == 'train':
                random.shuffle(lines)
            result = {'text': [], 'label': [], 'flag': []}
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in lines:
                data = json.loads(line)
                text = data['text']
                s_dict = {}  # sub : sub_type
                o_dict = {}  # obj : obj_type
                spo_dict = {}  # sub|obj : predicate|obj_type
                for spo in data['spo_list']:
                    sub = spo['subject']
                    # s_dict[spo['subject_type']] = spo['subject']
                    s_dict[spo['subject']] = spo['subject_type']
                    pre = spo['predicate']
                    p_o = pre + '|' + spo['object_type']['@value']
                    spo_dict[sub + '|' + spo['object']['@value']] = p_o
                    # o_dict[spo['object_type']['@value']] = spo['object']['@value']
                    o_dict[spo['object']['@value']] = spo['object_type']['@value']
                for sv, sk in s_dict.items():
                    for ov, ok in o_dict.items():
                        s_flag = self.s_entity_type[sk]  # '<s>, </s>'
                        o_flag = self.o_entity_type[ok]
                        s_start = self.search(text, sv)
                        s_end = s_start + len(sv)
                        text1 = text[:s_start] + s_flag[0] + sv + s_flag[1] + text[s_end:]
                        o_start = self.search(text1, ov)
                        o_end = o_start + len(ov)
                        text2 = text1[:o_start] + o_flag[0] + ov + o_flag[1] + text1[o_end:]
                        if sv + '|' + ov in spo_dict.keys():
                            labels = self.predicate2id[spo_dict[sv + '|' + ov]]
                        else:
                            labels = 0
                        for i in range(iter_num):
                            result['text'].append(text2)
                            result['label'].append(labels)
                            result['flag'].append((s_flag[0], o_flag[0]))
            return result

    def _load_schema(self, ):
        with open(self.schema_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            predicate_list = ["无关系"]
            s_entity = []
            o_entity = []
            pre_sub_obj = {}
            for line in lines:
                data = json.loads(line)
                if data['subject_type'] not in s_entity:
                    s_entity.append(data['subject_type'])
                if data['object_type'] not in o_entity:
                    o_entity.append(data['object_type'])
                predicate_list.append(data['predicate'] + '|' + data['object_type'])
                pre_sub_obj[data['predicate'] + '|' + data['object_type']] = [data['subject_type'], data['object_type']]

            s_entity_type = {}
            for i, e in enumerate(s_entity):  
                s_entity_type[e] = ('<s>', '</s>')  # unused4 unused5

            o_entity_type = {}
            for i, e in enumerate(o_entity):
                o_entity_type[e] = ('<o>', '</o>')

            predicate2id = {v: i for i, v in enumerate(predicate_list)}
            id2predicate = {i: v for i, v in enumerate(predicate_list)}

            self.pre_sub_obj = pre_sub_obj
            self.predicate2id = predicate2id
            self.id2predicate = id2predicate
            self.s_entity_type = s_entity_type
            self.o_entity_type = o_entity_type

    def search(self, sequence, pattern):
        """
        Find the substring pattern from the sequence
        If found, returns the first index; Otherwise return 0.
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return 0

    def build_text(self, data):
        text = data['text']
        result = []
        outputs = {'text': [], 'flag': [], "spo_list": []}
        for sub in data['sub_list']:
            for obj in data['obj_list']:
                if sub == obj:
                    continue
                sub_flag = ['<s>', '</s>']
                obj_flag = ['<o>', '</o>']
                sub_start = self.search(text, sub)  
                sub_end = sub_start + len(sub)
                text2 = text[:sub_start] + sub_flag[0] + sub + sub_flag[1] + text[sub_end:]
                obj_start = self.search(text2, obj)
                obj_end = obj_start + len(obj)
                text3 = text2[:obj_start] + obj_flag[0] + obj + obj_flag[1] + text2[obj_end:]
                result.append(
                    {'text': text3, 'flag': (sub_flag[0], obj_flag[0]), 'spo_list': {'subject': sub, 'object': obj}})
                outputs['text'].append(text3)
                outputs['flag'].append((sub_flag[0], obj_flag[0]))
                outputs['spo_list'].append({'subject': sub, 'object': obj})
        return result, outputs
   

class P2SODataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.train_path = os.path.join(root, 'CMeIE_train.json')
        self.dev_path = os.path.join(root, 'CMeIE_dev.json')
        self.test_path = os.path.join(root, 'CMeIE_test.json')
        self.gold_path = os.path.join(root, 'CMeIE_test.json')
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        self.num_labels = len(self.predicate2id.keys())
        self.args = args
        
    def get_train_sample(self):
        return self._pre_process(self.train_path, mode='train')

    def get_dev_sample(self):
        with jsonlines.open(self.dev_path, mode='r') as lines:
            samples = []
            for line in lines:
                samples.append(line)
        return samples

    def get_gold_sample(self):
        return self._pre_process(self.gold_path, mode='train')

    def get_test_sample(self, path):
        with jsonlines.open(path, mode='r') as lines:
            samples = []
            for line in lines:
                samples.append(line)
        return samples

    def merge(self,data_1,data_2):
        data_1.extend(data_2)
        return data_1
    
    def search(self, sequence, pattern):
        """
        Find the substring pattern from the sequence
        If found, returns the first index; Otherwise return 0.
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _pre_process(self, path, mode):
        new_data = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if mode == 'train':
                random.shuffle(lines)
            for line in lines:
                instance = json.loads(line)
                new_data.extend(self.build_data(instance, mode))
        return new_data
    
    def build_data(self, instance, mode='train'):
        result = []
        text = instance['text']
        p2so_dic = defaultdict(list)
        iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
        for spo in instance['spo_list']:
            sub = spo['subject']
            sub_type = spo['subject_type']
            obj = spo['object']['@value']
            obj_type = spo['object_type']['@value']
            p = spo['predicate'] + '|' + obj_type + '|' + sub_type
            if mode == 'test':
                p2so_dic[p] += [(sub, obj, spo['prob'])]
            else:
                p2so_dic[p] += [(sub, obj)]
        for p, so_list in p2so_dic.items():
            new_instance = {'text': text,
                            'p': p,
                            'so_list': so_list}
            for i in range(iter_num):
                result.append(new_instance)
        return result
        
    
    def _load_schema(self):
        with open(self.schema_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            predicate_list = ["无关系"]
            pre_sub_obj = {}
            for line in lines:
                data = json.loads(line)
                predicate_list.append(data['predicate'] + '|' + data['object_type'] + '|' + data['subject_type'])
                pre_sub_obj[data['predicate'] + '|' + data['object_type'] + '|' + data['subject_type']] = [data['subject_type'], data['object_type']]

            predicate2id = {v: i for i, v in enumerate(predicate_list)}
            id2predicate = {i: v for i, v in enumerate(predicate_list)}

            self.pre_sub_obj = pre_sub_obj
            self.predicate2id = predicate2id
            self.id2predicate = id2predicate
            
    def _extract_entity(self, start_logits, end_logits, text_start_id, text_end_id):
        threshold = self.args.predict_threshold
        # logits: seq
        start_ids = (start_logits[text_start_id:text_end_id] >= threshold).long()
        end_ids = (end_logits[text_start_id:text_end_id] >= threshold).long()

        start_end_tuple_list = []
        for i, start_id in enumerate(start_ids):
            if start_id == 0:
                continue  # Not the starting point
            if end_ids[i] == 1:  # The starting point and the ending point coincide
                start_end_tuple_list.append((i, i))
                continue
            j = i + 1
            find_end_tag = False
            while j < len(end_ids):
                if start_ids[j] == 1:
                    break  # Meet a new beginning before the end. Stop searching
                if end_ids[j] == 1:
                    start_end_tuple_list.append((i, j))
                    find_end_tag = True
                    break
                else:
                    j += 1
            if not find_end_tag:  # don't find the end->isolated point
                start_end_tuple_list.append((i, i))
        return start_end_tuple_list

    def extract_arg(self, start_logits, end_logits, text_start_id, text_end_id, text, text_mapping):
        arg_tuple = self._extract_entity(start_logits, end_logits, text_start_id, text_end_id)
        one_role_args = []
        for k in arg_tuple:
            # It doesn't seem to work
            if len(text_mapping) > 3:
                # len(text_mapping) : token size
                # k0: starting point    k1: end point
                start_split = text_mapping[k[0]]
                end_split = text_mapping[k[1]]
                if start_split != [] and end_split != []:
                    tmp = text[start_split[0]:end_split[-1] + 1]
                    one_role_args.append(tmp)
        return one_role_args
    
    def regular(self, spo):
        """
        Determine whether the spo complies with the rules
        return bool 
        """
        sub = spo['subject']
        if len(sub) == 1 and sub != '痔':
            return False
        return True
   
