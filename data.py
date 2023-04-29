# adapted from https://github.com/BlackNoodle/TUCORE-GCN

import json
import math
import os
import pickle
import random
import re
from collections import defaultdict
from itertools import permutations
import logging
from models.BERT import tokenization

import dgl
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


NUM_SPEAKER_DICT = {'DialogRE':10, 'DailyDialog':2, 'EmoryNLP':8, 'MELD':8, 'MRDA':26 }

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None, sample_id=None, answer_id=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label
        self.sample_id = sample_id
        self.answer_id = answer_id

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, speaker_ids, mention_ids, mentioned_h, mentioned_t, CLS_indices, hidialog_mask, transcript_id=0, answer_id=0):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.speaker_ids = speaker_ids
        self.mention_ids = mention_ids
        #add
        self.mentioned_h = mentioned_h
        self.mentioned_t = mentioned_t
        # add CLS
        self.CLS_indices = CLS_indices
        self.transcript_id = transcript_id
        self.answer_id = answer_id
        self.hidialog_mask = hidialog_mask

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class bertsProcessor(DataProcessor): #bert_s
    def __init__(self, src_file, n_class, data_name='DialogRE'):
        self.data_name = data_name
        def is_speaker(a):
            a = a.split()
            return len(a) == 2 and a[0] == "speaker" and a[1].isdigit()
        
        def rename(d, x, y):
            # they replace argument pair x,y by unused1, unused2 
            # and replace speaker_i by unused_j if speaker_i == x or y
            unused = ["[unused1]", "[unused2]"]
            a = []
            if is_speaker(x):
                a += [x]
            else:
                a += [None]
            if x != y and is_speaker(y):
                a += [y]
            else:
                a += [None]
            for i in range(len(a)):
                if a[i] is None:
                    continue
                d = d.replace(a[i] + ":", unused[i] + " :")
                if x == a[i]:
                    x = unused[i]
                if y == a[i]:
                    y = unused[i]
            return d, x, y
            
        random.seed(42)
        self.extra_info = 'MuTual' in str(src_file) or 'DDRel' in str(src_file) 
        self.D = [[], [], []]
        for sid in range(3):
            with open(src_file+["/train.json", "/dev.json", "/test.json"][sid], "r", encoding="utf8") as f:
                data = json.load(f)
            if sid == 0:
                random.shuffle(data)
            for i in range(len(data)):
                for j in range(len(data[i][1])): # argument pairs
                    if data_name in ['DDRel']: # each sample only has one label
                        rid = data[i][1][j]["rid"][0] - 1 # for now use -1 to correct label space to [0,num_label)
                    else:
                        rid = []
                        for k in range(n_class): # generate one-hot label vector
                            if k+1 in data[i][1][j]["rid"]:
                                rid += [1]
                            else:
                                rid += [0]
                    # replace speaker argument pair x,y by unused1, unused2 
                    d, h, t = rename('[TURNEND]\n'.join(data[i][0]).lower(), data[i][1][j]["x"].lower(), data[i][1][j]["y"].lower())
                    if self.extra_info:
                        d = [d,
                            h,
                            t,
                            rid,
                            data[i][1][j]["sample_id"],
                            data[i][1][j]["answer_id"]
                        ]
                    else:
                        d = [d,
                            h,
                            t,
                            rid]
                    self.D[sid] += [d]
        logger.info(str(len(self.D[0])) + "," + str(len(self.D[1])) + "," + str(len(self.D[2])))
        
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[0], "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[2], "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[1], "dev")

    def get_labels(self):
        """See base class."""
        return [str(x) for x in range(2)]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, d) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            text_a = data[i][0]
            text_b = data[i][1]
            text_c = data[i][2]
            if self.extra_info:
                sample_id = data[i][4]
                answer_id = data[i][5]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=data[i][3], text_c=text_c,
                sample_id=sample_id, answer_id=answer_id))
            else:
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=data[i][3], text_c=text_c))
            
        return examples

class bertsf1cProcessor(DataProcessor): #bert_s (conversational f1)
    def __init__(self, src_file, n_class):
        def is_speaker(a):
            a = a.split()
            return (len(a) == 2 and a[0] == "speaker" and a[1].isdigit())
        
        def rename(d, x, y):
            unused = ["[unused1]", "[unused2]"]
            a = []
            if is_speaker(x):
                a += [x]
            else:
                a += [None]
            if x != y and is_speaker(y):
                a += [y]
            else:
                a += [None]
            for i in range(len(a)):
                if a[i] is None:
                    continue
                d = d.replace(a[i] + ":", unused[i] + " :")
                if x == a[i]:
                    x = unused[i]
                if y == a[i]:
                    y = unused[i]
            return d, x, y
            
        random.seed(42)
        self.D = [[], [], []]
        for sid in range(1, 3):
            with open(src_file+["/dev.json", "/test.json"][sid-1], "r", encoding="utf8") as f:
                data = json.load(f)
            for i in range(len(data)):
                for j in range(len(data[i][1])):
                    rid = []
                    for k in range(n_class):
                        if k+1 in data[i][1][j]["rid"]:
                            rid += [1]
                        else:
                            rid += [0]
                    for l in range(1, len(data[i][0])+1):
                        d, h, t = rename('[TURNEND]\n'.join(data[i][0][:l]).lower(), data[i][1][j]["x"].lower(), data[i][1][j]["y"].lower())
                        d = [d,
                             h,
                             t,
                             rid]
                        self.D[sid] += [d]
        logger.info(str(len(self.D[0])) + "," + str(len(self.D[1])) + "," + str(len(self.D[2])))
        
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[0], "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[2], "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[1], "dev")

    def get_labels(self):
        """See base class."""
        return [str(x) for x in range(2)]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, d) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            text_a = data[i][0]
            text_b = data[i][1]
            text_c = data[i][2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=data[i][3], text_c=text_c))
            
        return examples


def tokenize(text, tokenizer, start_mention_id, head, tail, dataset):

    SPEAKER_IDS = ["speaker {}".format(x) for x in range(1,NUM_SPEAKER_DICT[dataset]+1)] # max_num_speaker
    speaker_dict = dict(zip(SPEAKER_IDS,range(1,NUM_SPEAKER_DICT[dataset]+1)))
    speaker_dict['[unused1]'] = NUM_SPEAKER_DICT[dataset]+1
    speaker_dict['[unused2]'] = NUM_SPEAKER_DICT[dataset]+2
    
    speaker2id = speaker_dict
    D = ['[unused1]', '[unused2]'] + SPEAKER_IDS
    text_tokens = []
    # textraw = [text]
    textraw = text.split('[turnend]\n')

    ntextraw = []
    for i, turn in enumerate(textraw):
        first_colon = turn.find(':')
        speakers = turn[:first_colon]
        dialog = turn[first_colon:]  
        speakers = [speakers]
        for delimiter in D:
            tmp_text = []
            for k in range(len(speakers)):
                tt = speakers[k].split(delimiter)
                for j,t in enumerate(tt):
                    tmp_text.append(t)
                    if j != len(tt)-1:
                        tmp_text.append(delimiter)
            speakers = tmp_text
        ntextraw.extend(speakers)
        ntextraw.append(dialog)
    textraw = ntextraw  

    text = []
    # speaker_ids, mention_ids may relate to speaker embedding in Fig 2 
    # mention_id is the order of apperance for a specific speaker_id 
    # both speaker_id and mention_id are assigned to each token in current turn
    speaker_ids = [] # same length as tokens
    mention_ids = [] # same length as tokens
    mention_id = start_mention_id
    speaker_id = 0 # number of mentioned speakers?
    mentioned_h = set()
    mentioned_t = set()

    for t in textraw:
        
        if t in SPEAKER_IDS:
            speaker_id = speaker2id[t]
            mention_id += 1
            
            # add [CLS] for each dialog
            text += ['[TURN]']
            speaker_ids.append(speaker_id)
            mention_ids.append(mention_id)

            tokens = tokenizer.tokenize(t+" ")
            for tok in tokens:
                text += [tok]
                speaker_ids.append(speaker_id)
                mention_ids.append(mention_id)

        elif t in ['[unused1]', '[unused2]']:
            speaker_id = speaker2id[t]
            mention_id += 1

            # add [CLS] for each dialog
            text += ['[TURN]']
            speaker_ids.append(speaker_id)
            mention_ids.append(mention_id)
            
            text += [t]
            speaker_ids.append(speaker_id)
            mention_ids.append(mention_id)
        else:
            tokens = tokenizer.tokenize(t)

            for tok in tokens:

                text += [tok]
                speaker_ids.append(speaker_id)
                mention_ids.append(mention_id)

        # establish an edge between an argument and a turn that mentioned it
        if head in t:
            mentioned_h.add(mention_id)
        if tail in t:
            mentioned_t.add(mention_id)



    return text, speaker_ids, mention_ids, mentioned_h, mentioned_t


def tokenize2(text, tokenizer, dataset):

    SPEAKER_IDS = ["speaker {}".format(x) for x in range(1,NUM_SPEAKER_DICT[dataset]+1)] # max_num_speaker
    speaker_dict = dict(zip(SPEAKER_IDS,range(1,NUM_SPEAKER_DICT[dataset]+1)))
    speaker_dict['[unused1]'] = NUM_SPEAKER_DICT[dataset]+1
    speaker_dict['[unused2]'] = NUM_SPEAKER_DICT[dataset]+2
    
    speaker2id = speaker_dict
    D = ['[unused1]', '[unused2]'] + SPEAKER_IDS
    text_tokens = []
    textraw = [text]
    for delimiter in D:
        ntextraw = []
        for i in range(len(textraw)):
            t = textraw[i].split(delimiter)
            for j in range(len(t)):
                ntextraw += [t[j]]
                if j != len(t)-1:
                    ntextraw += [delimiter]
        textraw = ntextraw
    text = []
    speaker_ids = []
    speaker_id = 0

    # add 
    mentioned_h = set()
    mentioned_t = set()
    
    # add [CLS] for each dialog
    text += ['[TURN]']
    speaker_ids.append(0)
    for t in textraw:

        # if t in ['speaker 1', 'speaker 2', 'speaker 3', 'speaker 4', 'speaker 5', 'speaker 6', 'speaker 7', 'speaker 8', 'speaker 9']:
        if t in SPEAKER_IDS:

            speaker_id = speaker2id[t]

            tokens = tokenizer.tokenize(t+" ")
            for tok in tokens:
                text += [tok]
                speaker_ids.append(speaker_id)
        elif t in ['[unused1]', '[unused2]']:
            speaker_id = speaker2id[t]

            text += [t]
            speaker_ids.append(speaker_id)
        else:

            tokens = tokenizer.tokenize(t)
            for tok in tokens:
                text += [tok]
                speaker_ids.append(speaker_id)

    return text, speaker_ids

def convert_examples_to_features(examples, max_seq_length, tokenizer, dataset='DialogRE'):
    """Loads a data file into a list of `InputBatch`s."""

    print("#examples", len(examples))

    features = [[]]
    for (ex_index, example) in enumerate(examples):
        # add h,t
        h, t = example.text_b, example.text_c

        tokens_a, tokens_a_speaker_ids, tokens_a_mention_ids, mentioned_h, mentioned_t = tokenize(example.text_a, tokenizer, 0, h, t, dataset)
        
        tokens_b, tokens_b_speaker_ids = tokenize2(example.text_b, tokenizer, dataset)
        tokens_c, tokens_c_speaker_ids = tokenize2(example.text_c, tokenizer, dataset)
        transcript_id, answer_id = 0, 0
        if dataset in ['MuTual', 'DDRel']:
            transcript_id = example.sample_id
            answer_id = example.answer_id

        tokens_a, tokens_a_speaker_ids, tokens_a_mention_ids, popped_mention_id, popped_min = _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4, tokens_a_speaker_ids, tokens_b_speaker_ids, tokens_c_speaker_ids, tokens_a_mention_ids)
        mentioned_h = set([x-popped_min+1 for x in mentioned_h if x not in popped_mention_id])
        mentioned_t = set([x-popped_min+1 for x in mentioned_t if x not in popped_mention_id])
        tokens_b_mention_ids = [max(tokens_a_mention_ids) + 1 for _ in range(len(tokens_b))]
        tokens_c_mention_ids = [max(tokens_a_mention_ids) + 2 for _ in range(len(tokens_c))]

        tokens_b = tokens_b + ["[SEP]"] + tokens_c
        tokens_b_speaker_ids = tokens_b_speaker_ids + [0] + tokens_c_speaker_ids
        tokens_b_mention_ids = tokens_b_mention_ids + [0] + tokens_c_mention_ids

        tokens = []
        segment_ids = []
        speaker_ids = []
        mention_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        speaker_ids.append(0)
        mention_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        speaker_ids = speaker_ids + tokens_a_speaker_ids
        mention_ids = mention_ids + tokens_a_mention_ids
        tokens.append("[SEP]")
        segment_ids.append(0)
        speaker_ids.append(0)
        mention_ids.append(0)

        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        speaker_ids = speaker_ids + tokens_b_speaker_ids
        mention_ids = mention_ids + tokens_b_mention_ids
        tokens.append("[SEP]")
        segment_ids.append(1)
        speaker_ids.append(0)
        mention_ids.append(0)

        # add CLS for each turn, get their indices
        CLS_indices = list(map(lambda x:  x == '[TURN]', tokens))
        # tokens[tokens=='TURN'] = '</s>'
        tokens = list(map(lambda t: t if t != '[TURN]' else '[CLS]',tokens))     
        CLS_indices[0] = True

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        len_tokens = len(input_ids)
        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            speaker_ids.append(0)
            mention_ids.append(0)
            CLS_indices.append(False)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(speaker_ids) == max_seq_length
        assert len(mention_ids) == max_seq_length
        # assert len(CLS_indices) == max(mention_ids) + 1

        label_id = example.label 
        slen = max_seq_length
        CLS_attn_mask = np.full((slen,slen),0)
        head_cls = 1
        # len_tokens = (input_mask == 1).sum()
        for j, cls in enumerate(CLS_indices):
            if j > 1 and cls:
                CLS_attn_mask[head_cls, head_cls:j] = 1
                head_cls = j
            elif not cls or j == 0:
                CLS_attn_mask[j, :len_tokens] = 1

        CLS_attn_mask[head_cls, head_cls:len_tokens] = 1
        
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [x for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info(
                    "speaker_ids: %s" % " ".join([str(x) for x in speaker_ids]))
            logger.info(
                    "mention_ids: %s" % " ".join([str(x) for x in mention_ids]))
            logger.info(
                    "special_tokens_masked: %s" % " ".join([str(x) for x in CLS_indices]))       

        features[-1].append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id,
                        speaker_ids=speaker_ids,
                        mention_ids=mention_ids,
                        mentioned_h=mentioned_h,
                        mentioned_t=mentioned_t,
                        CLS_indices=CLS_indices,
                        transcript_id=transcript_id,
                        answer_id=answer_id,
                        hidialog_mask=CLS_attn_mask
                        )) # add h,t CLS indices
        if len(features[-1]) == 1:
            features.append([])

    if len(features[-1]) == 0:
        features = features[:-1]
    print('#features', len(features))
    return features

def convert_examples_to_features_roberta(examples, max_seq_length, tokenizer, dataset='DialogRE'):
    """Loads a data file into a list of `InputBatch`s."""

    print("#examples", len(examples))

    features = [[]]
    for (ex_index, example) in enumerate(examples):
        
        # add h,t 
        h, t = example.text_b, example.text_c
        tokens_a, tokens_a_speaker_ids, tokens_a_mention_ids, mentioned_h, mentioned_t = tokenize(example.text_a, tokenizer, 0, h, t, dataset)

        tokens_b, tokens_b_speaker_ids = tokenize2(example.text_b, tokenizer, dataset)

        tokens_c, tokens_c_speaker_ids = tokenize2(example.text_c, tokenizer, dataset)
        transcript_id, answer_id = 0, 0
        if dataset in ['MuTual', 'DDRel']:
            transcript_id = example.sample_id
            answer_id = example.answer_id

        tokens_a, tokens_a_speaker_ids, tokens_a_mention_ids, popped_mention_id, popped_min  = _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 6, tokens_a_speaker_ids, tokens_b_speaker_ids, tokens_c_speaker_ids, tokens_a_mention_ids)#, turn_no=transcript_id)
        mentioned_h = set([x-popped_min+1 for x in mentioned_h if x not in popped_mention_id])
        mentioned_t = set([x-popped_min+1 for x in mentioned_t if x not in popped_mention_id])
        tokens_b_mention_ids = [max(tokens_a_mention_ids) + 1 for _ in range(len(tokens_b))]
        tokens_c_mention_ids = [max(tokens_a_mention_ids) + 2 for _ in range(len(tokens_c))]

        tokens_b = tokens_b + ['</s>', '</s>'] + tokens_c
        tokens_b_speaker_ids = tokens_b_speaker_ids + [0, 0] + tokens_c_speaker_ids
        tokens_b_mention_ids = tokens_b_mention_ids + [0, 0] + tokens_c_mention_ids

        tokens = []
        segment_ids = []
        speaker_ids = []
        mention_ids = []
        tokens.append('<s>')
        segment_ids.append(0)
        speaker_ids.append(0)
        mention_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        speaker_ids = speaker_ids + tokens_a_speaker_ids
        mention_ids = mention_ids + tokens_a_mention_ids
        tokens.append('</s>')
        segment_ids.append(0)
        speaker_ids.append(0)
        mention_ids.append(0)
        tokens.append('</s>')
        segment_ids.append(0)
        speaker_ids.append(0)
        mention_ids.append(0)

        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        speaker_ids = speaker_ids + tokens_b_speaker_ids
        mention_ids = mention_ids + tokens_b_mention_ids
        tokens.append('</s>')
        segment_ids.append(1)
        speaker_ids.append(0)
        mention_ids.append(0)

        # add CLS for each turn, get their indices
        CLS_indices = list(map(lambda x:  x == '[TURN]', tokens))
        # tokens[tokens=='TURN'] = '</s>'
        tokens = list(map(lambda t: t if t != '[TURN]' else '</s>',tokens))     
        CLS_indices[0] = True
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        len_tokens = len(input_ids)
        assert len(input_ids) == len(input_mask)
        assert len(input_mask) == len(segment_ids)
        assert len(segment_ids) == len(speaker_ids)
        assert len(speaker_ids) == len(mention_ids)
        assert len(mention_ids) == len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(1)
            input_mask.append(0)
            segment_ids.append(0)
            speaker_ids.append(0)
            mention_ids.append(0)
            CLS_indices.append(False)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(speaker_ids) == max_seq_length
        assert len(mention_ids) == max_seq_length


        label_id = example.label 
        slen = max_seq_length
        CLS_attn_mask = np.full((slen,slen),0)
        head_cls = 1
        for j, cls in enumerate(CLS_indices):
            if j > 1 and cls:
                CLS_attn_mask[head_cls, head_cls:j] = 1
                head_cls = j
            elif not cls or j == 0:
                CLS_attn_mask[j, :len_tokens] = 1

        CLS_attn_mask[head_cls, head_cls:len_tokens] = 1  
        
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [x for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info(
                    "speaker_ids: %s" % " ".join([str(x) for x in speaker_ids]))
            logger.info(
                    "mention_ids: %s" % " ".join([str(x) for x in mention_ids]))
            logger.info(
                    "special_tokens_masked: %s" % " ".join([str(x) for x in CLS_indices]))                    

        features[-1].append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id,
                        speaker_ids=speaker_ids,
                        mention_ids=mention_ids,
                        mentioned_h=mentioned_h,
                        mentioned_t=mentioned_t,
                        CLS_indices=CLS_indices,
                        transcript_id=transcript_id,
                        answer_id=answer_id,
                        hidialog_mask=CLS_attn_mask
                        )) # add h,t CLS indices

        if len(features[-1]) == 1:
            features.append([])

    if len(features[-1]) == 0:
        features = features[:-1]
    print('#features', len(features))
    return features

def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length, tokens_a_speaker_ids, tokens_b_speaker_ids, tokens_c_speaker_ids, tokens_a_mention_ids, turn_no=None):
    """Truncates a sequence tuple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    if turn_no is not None:
        turn_start_indices = [i for i,x in enumerate(tokens_a) if x == '<s>']
        turn_end_indices = ([i-1 for i,x in enumerate(tokens_a) if x == '<s>'] + [len(tokens_a)-1])[1:]
        # turn_end_indices = turn_end_indices[2:]
        turn_start_idx = turn_start_indices[turn_no], turn_end_indices[turn_no]
        l, r = 0, len(turn_start_indices) - 1
        candidate_indices = []
        while l < r:
            mid = (l+r) // 2
            if mid <= turn_no:
                candidate_indices.extend(range(turn_end_indices[l],turn_start_indices[l]-1,-1))
                l += 1
            if mid >= turn_no:
                candidate_indices.extend(range(turn_end_indices[r],turn_start_indices[r]-1,-1))
                r -= 1
    else:
        candidate_indices = [x for x in range(len(tokens_a)-1,-1,-1)]
    popped_mention_id = set()
    tokens_a_mention_ids_copy = set(tokens_a_mention_ids.copy())
    indices_to_pop = []
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break

        indices_to_pop.append(candidate_indices.pop(0))
        max_length += 1 
    
    tokens_a = [token for i, token in enumerate(tokens_a) if i not in indices_to_pop]
    tokens_a_speaker_ids = [x for i, x in enumerate(tokens_a_speaker_ids) if i not in indices_to_pop]
    tokens_a_mention_ids = [x for i, x in enumerate(tokens_a_mention_ids) if i not in indices_to_pop]

    popped_min, popped_max = min(tokens_a_mention_ids), max(tokens_a_mention_ids)
    popped_mention_id = [x for x in tokens_a_mention_ids_copy if x not in set(tokens_a_mention_ids)]
    if popped_min != min(tokens_a_mention_ids_copy):
        tokens_a_mention_ids = [x - popped_min + 1 for x in tokens_a_mention_ids]

    return tokens_a, tokens_a_speaker_ids, tokens_a_mention_ids, popped_mention_id, popped_min
def get_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor



class HiDialogDataset(IterableDataset):

    def __init__(self, src_file, save_file, max_seq_length, tokenizer, n_class, encoder_type, dataset='DialogRE'):

        super(HiDialogDataset, self).__init__()

        self.data = None
        self.input_max_length = max_seq_length
        self.dataset = dataset

        print('Reading data from {}.'.format(src_file))
        if os.path.exists(save_file):
            with open(file=save_file, mode='rb') as fr:
                info = pickle.load(fr)
                self.data = info['data']
            print('load preprocessed data from {}.'.format(save_file))

        else:
            self.data = []

            bertsProcessor_class = bertsProcessor(src_file, n_class, dataset)

            if "train" in save_file:
                examples = bertsProcessor_class.get_train_examples(save_file)
            elif "dev" in save_file:
                examples = bertsProcessor_class.get_dev_examples(save_file)
            elif "test" in save_file:
                examples = bertsProcessor_class.get_test_examples(save_file)

            else:
                print(error)
            if encoder_type in ["BERT","DeBERTa","Electra"]:
                features = convert_examples_to_features(examples, max_seq_length, tokenizer, dataset=self.dataset)
            else:
                features = convert_examples_to_features_roberta(examples, max_seq_length, tokenizer, dataset=self.dataset)

            for f in features:


                entity_edges_infor = {'h':f[0].mentioned_h, 't':f[0].mentioned_t}
                speaker_infor = self.make_speaker_infor(f[0].speaker_ids, f[0].mention_ids)

                turn_node_num = max(f[0].mention_ids) - 2
                head_mention_id = max(f[0].mention_ids) - 1
                tail_mention_id = max(f[0].mention_ids)
                entity_edges_infor['h'].add(head_mention_id)
                entity_edges_infor['t'].add(tail_mention_id)


                graph, used_mention = self.create_graph(speaker_infor, turn_node_num, entity_edges_infor, head_mention_id, tail_mention_id)

                assert len(used_mention) == (max(f[0].mention_ids) + 1)

                self.data.append({
                    'input_ids': np.array(f[0].input_ids),
                    'segment_ids': np.array(f[0].segment_ids),
                    'input_mask': np.array(f[0].input_mask),
                    'speaker_ids': np.array(f[0].speaker_ids),
                    'label_ids': f[0].label_id if self.dataset in ['DDRel'] else np.array(f[0].label_id),
                    'mention_id': np.array(f[0].mention_ids),
                    'graph': graph,
                    'cls_indices':np.array(f[0].CLS_indices,dtype=bool),
                    'transcript_ids':np.array(f[0].transcript_id),
                    'answer_ids': np.array(f[0].answer_id),
                    'hidialog_mask':np.array(f[0].hidialog_mask)
                    })
                # i += 1
            # # save data
            # with open(file=save_file, mode='wb') as fw:
            #     pickle.dump({'data': self.data}, fw)
            # print('mention not the same',cnt)
            print('finish reading {} and save preprocessed data to {}.'.format(src_file, save_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)
    
    def turn2speaker(self, turn):
        return turn.split()[1]
    
    def make_speaker_infor(self, speaker_id, mention_id):
        tmp = defaultdict(set)
        for i in range(1, len(speaker_id)):
            if speaker_id[i] == 0:
                break
            tmp[speaker_id[i]].add(mention_id[i])
        
        speaker_infor = dict()
        for k, va in tmp.items():
            speaker_infor[k] = list(va)
        return speaker_infor

    def create_graph(self, speaker_infor, turn_node_num, entity_edges_infor, head_mention_id, tail_mention_id):

        # used_mention is used for sanity check
        d = defaultdict(list)
        used_mention = set()
        used_mention.add(0)
        for _, mentions in speaker_infor.items():
            for h, t in permutations(mentions, 2):
                d[('node', 'speaker', 'node')].append((h, t))
                used_mention.add(h)
                used_mention.add(t)
                
        if d[('node', 'speaker', 'node')] == []:
            d[('node', 'speaker', 'node')].append((1, 0))
            used_mention.add(1)
            used_mention.add(0)
        
        # add dialog edges for HAN v3 v5
        for i in range(1, turn_node_num+1):
            d[('node', 'dialog', 'node')].append((i, 0))
            d[('node', 'dialog', 'node')].append((0, i))
            used_mention.add(i)
            used_mention.add(0)
        if d[('node', 'dialog', 'node')] == []:
            d[('node', 'dialog', 'node')].append((1, 0))
            used_mention.add(1)
            used_mention.add(0)

        # entity -> subject
        # add entity edges
        for mention in entity_edges_infor['h']:
            d[('node', 'entity', 'node')].append((head_mention_id, mention))
            d[('node', 'entity', 'node')].append((mention, head_mention_id))
            used_mention.add(head_mention_id)
            used_mention.add(mention)
                
        # entity -> object
        for mention in entity_edges_infor['t']:
            d[('node', 'entity', 'node')].append((tail_mention_id, mention))
            d[('node', 'entity', 'node')].append((mention, tail_mention_id))
            used_mention.add(tail_mention_id)
            used_mention.add(mention)

        
        # if an entity is not mentioned in any turns
        # then connect it to all turns
        if entity_edges_infor['h'] == []:
            for i in range(1, turn_node_num+1):
                d[('node', 'entity', 'node')].append((head_mention_id, i))
                d[('node', 'entity', 'node')].append((i, head_mention_id))
                used_mention.add(head_mention_id)
        if entity_edges_infor['t'] == []:
            for i in range(1, turn_node_num+1):
                d[('node', 'entity', 'node')].append((tail_mention_id, i))
                d[('node', 'entity', 'node')].append((i, tail_mention_id))
                used_mention.add(tail_mention_id)

        # add sequence edges 
        # if there is only one turn in a dialogue, sequence edge won't exist
        # do nothing in this case 
        turns = [i for i in range(1, turn_node_num+1)]
        if len(turns) >= 2:
            for h, t in permutations(turns, 2):
                d[('node', 'sequence', 'node')].append((h, t))
                used_mention.add(h)                
                used_mention.add(t)
        
        if d[('node', 'sequence', 'node')] == []:
            d[('node', 'sequence', 'node')].append((1, 0))
            used_mention.add(1)
            used_mention.add(0)
        

        graph = dgl.heterograph(d)

        return graph, used_mention

class HiDialogDataset4f1c(IterableDataset):

    def __init__(self, src_file, save_file, max_seq_length, tokenizer, n_class, encoder_type):

        super(HiDialogDataset4f1c, self).__init__()

        self.data = None
        self.input_max_length = max_seq_length

        print('Reading data from {}.'.format(src_file))
        if os.path.exists(save_file):
            with open(file=save_file, mode='rb') as fr:
                info = pickle.load(fr)
                self.data = info['data']
            print('load preprocessed data from {}.'.format(save_file))

        else:
            self.data = []

            bertsProcessor_class = bertsf1cProcessor(src_file, n_class)
            if "dev" in save_file:
                examples = bertsProcessor_class.get_dev_examples(save_file)
            elif "test" in save_file:
                examples = bertsProcessor_class.get_test_examples(save_file)
            else:
                print(error)
            if encoder_type in ["BERT","DeBERTa","Electra"]:
                features = convert_examples_to_features(examples, max_seq_length, tokenizer)
            else:
                features = convert_examples_to_features_roberta(examples, max_seq_length, tokenizer)

            
            for f in features:
                entity_edges_infor = {'h':f[0].mentioned_h, 't':f[0].mentioned_t}
                speaker_infor = self.make_speaker_infor(f[0].speaker_ids, f[0].mention_ids)
                turn_node_num = max(f[0].mention_ids) - 2
                head_mention_id = max(f[0].mention_ids) - 1
                tail_mention_id = max(f[0].mention_ids)
                entity_edges_infor['h'].add(head_mention_id)
                entity_edges_infor['t'].add(tail_mention_id)
                graph, used_mention = self.create_graph(speaker_infor, turn_node_num, entity_edges_infor, head_mention_id, tail_mention_id)
                assert len(used_mention) == (max(f[0].mention_ids) + 1)

                self.data.append({
                    'input_ids': np.array(f[0].input_ids),
                    'segment_ids': np.array(f[0].segment_ids),
                    'input_mask': np.array(f[0].input_mask),
                    'speaker_ids': np.array(f[0].speaker_ids),
                    'label_ids': np.array(f[0].label_id),
                    'mention_id': np.array(f[0].mention_ids),
                    'graph': graph,
                    'cls_indices':np.array(f[0].CLS_indices,dtype=bool),
                    'hidialog_mask':np.array(f[0].hidialog_mask)

                    })

            # with open(file=save_file, mode='wb') as fw:
            #     pickle.dump({'data': self.data}, fw)
            print('finish reading {} and save preprocessed data to {}.'.format(src_file, save_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)
    
    def turn2speaker(self, turn):
        return turn.split()[1]
    
    def make_speaker_infor(self, speaker_id, mention_id):
        tmp = defaultdict(set)
        for i in range(1, len(speaker_id)):
            if speaker_id[i] == 0:
                break
            tmp[speaker_id[i]].add(mention_id[i])
        
        speaker_infor = dict()
        for k, va in tmp.items():
            speaker_infor[k] = list(va)
        return speaker_infor

    def create_graph(self, speaker_infor, turn_node_num, entity_edges_infor, head_mention_id, tail_mention_id):

        # used_mention is used for sanity check
        d = defaultdict(list)
        used_mention = set()
        used_mention.add(0)
        for _, mentions in speaker_infor.items():
            for h, t in permutations(mentions, 2):
                d[('node', 'speaker', 'node')].append((h, t))
                used_mention.add(h)
                used_mention.add(t)
                
        if d[('node', 'speaker', 'node')] == []:
            d[('node', 'speaker', 'node')].append((1, 0))
            used_mention.add(1)
            used_mention.add(0)
        
        # add dialog edges for HAN v3 v5
        for i in range(1, turn_node_num+1):
            d[('node', 'dialog', 'node')].append((i, 0))
            d[('node', 'dialog', 'node')].append((0, i))
            used_mention.add(i)
            used_mention.add(0)
        if d[('node', 'dialog', 'node')] == []:
            d[('node', 'dialog', 'node')].append((1, 0))
            used_mention.add(1)
            used_mention.add(0)

        # entity -> subject
        # add entity edges
        for mention in entity_edges_infor['h']:
            d[('node', 'entity', 'node')].append((head_mention_id, mention))
            d[('node', 'entity', 'node')].append((mention, head_mention_id))
            used_mention.add(head_mention_id)
            used_mention.add(mention)
                
        # entity -> object
        for mention in entity_edges_infor['t']:
            d[('node', 'entity', 'node')].append((tail_mention_id, mention))
            d[('node', 'entity', 'node')].append((mention, tail_mention_id))
            used_mention.add(tail_mention_id)
            used_mention.add(mention)

        
        # if an entity is not mentioned in any turns
        # then connect it to all turns
        if entity_edges_infor['h'] == []:
            for i in range(1, turn_node_num+1):
                d[('node', 'entity', 'node')].append((head_mention_id, i))
                d[('node', 'entity', 'node')].append((i, head_mention_id))
                used_mention.add(head_mention_id)
        if entity_edges_infor['t'] == []:
            for i in range(1, turn_node_num+1):
                d[('node', 'entity', 'node')].append((tail_mention_id, i))
                d[('node', 'entity', 'node')].append((i, tail_mention_id))
                used_mention.add(tail_mention_id)

        # add sequence edges 
        # if there is only one turn in a dialogue, sequence edge won't exist
        # do nothing in this case 
        turns = [i for i in range(1, turn_node_num+1)]
        if len(turns) >= 2:
            for h, t in permutations(turns, 2):
                d[('node', 'sequence', 'node')].append((h, t))
                used_mention.add(h)                
                used_mention.add(t)
        
        if d[('node', 'sequence', 'node')] == []:
            d[('node', 'sequence', 'node')].append((1, 0))
            used_mention.add(1)
            used_mention.add(0)
        

        graph = dgl.heterograph(d)

        return graph, used_mention

class HiDialogDataloader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False, relation_num=36, max_length=512, dataset_name='DialogRE'):
        super(HiDialogDataloader, self).__init__(dataset, batch_size=batch_size, worker_init_fn=np.random.seed(42))
        self.shuffle = shuffle
        self.length = len(self.dataset)
        self.max_length = max_length
        
        self.relation_num = relation_num
        if dataset_name == 'MuTual':
            assert batch_size % 4 == 0
            self.order = [[i*4+j for j in range(4)] for i in range(self.length // 4)]
        else:
            self.order = list(range(self.length))
        self.dataset_name = dataset_name

    def __iter__(self):
        # shuffle
        if self.shuffle:
            random.shuffle(self.order)
            order = self.order if self.dataset_name != 'MuTual' else [idx for group in self.order for idx in group]
            self.data = [self.dataset[idx] for idx in order]
            batch_num = self.length // self.batch_size
        else:
            self.data = self.dataset
            batch_num = math.ceil(self.length / self.batch_size)

        self.batches = [self.data[idx * self.batch_size: min(self.length, (idx + 1) * self.batch_size)]
                        for idx in range(0, batch_num)]
        self.batches_order = [self.order[idx * self.batch_size: min(self.length, (idx + 1) * self.batch_size)]
                              for idx in range(0, batch_num)]

        # begin
        input_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        segment_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        input_mask = torch.LongTensor(self.batch_size, self.max_length).cpu()
        mention_id = torch.LongTensor(self.batch_size, self.max_length).cpu()
        speaker_id = torch.LongTensor(self.batch_size, self.max_length).cpu()

        if self.dataset_name in ['DDRel']:
            label_ids = torch.LongTensor(self.batch_size)
        else:
            label_ids = torch.Tensor(self.batch_size, self.relation_num).cpu()

        # add CLS
        cls_indices = torch.Tensor(self.batch_size, self.max_length).bool().cpu()
        transcript_ids = np.zeros(self.batch_size)
        answer_ids = np.zeros(self.batch_size)

        hidialog_masks = torch.FloatTensor(self.batch_size, self.max_length, self.max_length).cpu()

        for idx, minibatch in enumerate(self.batches):
            cur_bsz = len(minibatch)

            for mapping in [input_ids, segment_ids, input_mask, mention_id, label_ids, speaker_id]: # remove turn mask
                if mapping is not None:
                    mapping.zero_()

            graph_list = []

            for i, example in enumerate(minibatch): # remove turn mask
                if 'transcript_ids' in example.keys() and 'answer_ids' in example.keys():
                    mini_input_ids, mini_segment_ids, mini_input_mask, mini_label_ids, mini_mention_id, mini_speaker_id, graph, mini_cls_indices, mini_transcript_ids, mini_answer_ids, mini_hidialog_mask = \
                        example['input_ids'], example['segment_ids'], example['input_mask'], example['label_ids'], \
                        example['mention_id'], example['speaker_ids'],  example['graph'], example['cls_indices'], example['transcript_ids'], example['answer_ids'], example['hidialog_mask']
                else:
                    mini_input_ids, mini_segment_ids, mini_input_mask, mini_label_ids, mini_mention_id, mini_speaker_id, graph, mini_cls_indices, mini_transcript_ids, mini_answer_ids, mini_hidialog_mask = \
                        example['input_ids'], example['segment_ids'], example['input_mask'], example['label_ids'], \
                        example['mention_id'], example['speaker_ids'],  example['graph'], example['cls_indices'], 0, 0, example['hidialog_mask']

                graph_list.append(graph.to(torch.device('cuda:0')))

                word_num = mini_input_ids.shape[0]
                
                input_ids[i, :word_num].copy_(torch.from_numpy(mini_input_ids))
                segment_ids[i, :word_num].copy_(torch.from_numpy(mini_segment_ids))
                input_mask[i, :word_num].copy_(torch.from_numpy(mini_input_mask))
                mention_id[i, :word_num].copy_(torch.from_numpy(mini_mention_id))
                speaker_id[i, :word_num].copy_(torch.from_numpy(mini_speaker_id))

                if self.dataset_name in ['DDRel']:
                    label_ids[i] = int(mini_label_ids)
                else:
                    relation_num = mini_label_ids.shape[0] 
                    label_ids[i, :relation_num].copy_(torch.from_numpy(mini_label_ids))
                # add CLS
                cls_indices[i, :word_num].copy_(torch.from_numpy(mini_cls_indices))
                hidialog_masks[i, :word_num, :word_num].copy_(torch.from_numpy(mini_hidialog_mask))
                transcript_ids[i] = mini_transcript_ids
                answer_ids[i] = mini_answer_ids

            context_word_mask = input_mask > 0
            context_word_length = context_word_mask.sum(1)
            batch_max_length = context_word_length.max()
            # batch_max_length = self.max_length
            yield {'input_ids': get_cuda(input_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'segment_ids': get_cuda(segment_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'input_masks': get_cuda(input_mask[:cur_bsz, :batch_max_length].contiguous()),
                   'mention_ids': get_cuda(mention_id[:cur_bsz, :batch_max_length].contiguous()),
                   'speaker_ids': get_cuda(speaker_id[:cur_bsz, :batch_max_length].contiguous()),
                   'label_ids':  get_cuda(label_ids[:cur_bsz].contiguous()) \
                                    if self.dataset_name in ['DDRel'] else  get_cuda(label_ids[:cur_bsz, :self.relation_num].contiguous()),
                   'graphs': graph_list,
                   'cls_indices': get_cuda(cls_indices[:cur_bsz, :batch_max_length].contiguous()), # add CLS
                   'transcript_ids': transcript_ids, #transcript_ids if self.dataset_name != 'DialogRE' else
                   'answer_ids': answer_ids,
                   'hidialog_masks':get_cuda(hidialog_masks[:cur_bsz, :batch_max_length, :batch_max_length].contiguous())
                   }
