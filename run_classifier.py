# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
import pickle
from tqdm import tqdm, trange

import numpy as np
import torch
import time

from models.BERT import tokenization
from models.BERT.HiDialog_BERT import BertConfig, HiDialog_BERT

from models.RoBERTa.tokenization_roberta import RobertaTokenizer
from models.RoBERTa.HiDialog_RoBERTa import HiDialog_RoBERTa
from models.RoBERTa.configuration_roberta import RobertaConfig
from optimization import BERTAdam
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

import json
import re
import os
import copy

from data import HiDialogDataset, HiDialogDataloader, HiDialogDataset4f1c

n_classes = {
        "DialogRE": 36,
        "MELD": 7,
        "EmoryNLP": 7,
        "DailyDialog": 7,
        "MuTual": 1,
        "MRDA": 5,
        "DDRel": 13 # default = 13  
    }
# the reason set #class of mutual it to 1 is 
# to align with other dataset, where there is a 
# outlier class(meaning no label), see line 124-129 in data.py
reverse_order = False
sa_step = False


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# torch.autograd.set_detect_anomaly(True)
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def calc_test_result(logits, labels_all, data_name):
    true_label=[]
    predicted_label=[]
    logits = np.asarray(logits)
    logits = list(1 / (1 + np.exp(-logits)))

    for i in range(len(logits)):
        true_label.append(np.argmax(labels_all[i]))
        predicted_label.append(np.argmax(logits[i]))

    print("Confusion Matrix :")
    print(confusion_matrix(true_label, predicted_label))
    print("Classification Report :")
    if data_name == "DailyDialog":
        print(classification_report(true_label, predicted_label, labels=[1,2,3,4,5,6], digits=4))
        p_weighted, r_weighted, f_weighted, support_weighted = precision_recall_fscore_support(true_label, predicted_label, labels=[1,2,3,4,5,6], average='micro')
    else:
        print(classification_report(true_label, predicted_label, digits=4))
        p_weighted, r_weighted, f_weighted, support_weighted = precision_recall_fscore_support(true_label, predicted_label, average='weighted')
    print('Weighted FScore: \n ', p_weighted, r_weighted, f_weighted, support_weighted)
    return p_weighted, r_weighted, f_weighted, support_weighted

def accuracy(out, labels):
    out = out.reshape(-1)
    out = 1 / (1 + np.exp(-out))
    return np.sum((out > 0.5) == (labels > 0.5)) / 36

def accuracy_dac(logits, labels):
    true_label=[]
    predicted_label=[]
    true_label = np.where(labels==1)[1]
    predicted_label = np.argmax(logits,axis=1)

    return np.sum(true_label==predicted_label)

def calc_test_result_mutual(sample_ids, predictions, labels, answer_ids=None):
    idx2answer = ['A','B','C','D']
    samples = {}
    r1, r2, mrr = 0, 0, 0
    is_test = answer_ids is not None 
    answer_ids = copy.deepcopy(sample_ids) if answer_ids is None else answer_ids
    for sample_id, pred, label, answer_id in zip(sample_ids, predictions, labels, answer_ids):
        samples.setdefault(sample_id, []).append([label, pred, answer_id])

    if is_test:
        pred2save = []
        samples = dict(sorted(samples.items()))
        for sample_id, predictions in samples.items():
            line = '\t'.join(['test_'+str(int(sample_id))] + \
            [idx2answer[int(x[-1])] for x in sorted(predictions, key=lambda x: x[1], reverse=True)])
            pred2save.append(line+'\n')
        return pred2save
    else:
        for sample_id, predictions in samples.items():
            predictions = [x[0] for x in sorted(predictions, key=lambda x: x[1], reverse=True)]
            index = predictions.index(1)
            if index == 0:
                r1 += 1
            elif index == 1:
                r2 += 1
            mrr += 1 / (index + 1)
        total = len(list(samples.keys()))
        r1, r2, mrr = r1/total, (r1+r2)/total, mrr/total
        print('Results For MuTual:\nR@1: {} R@2: {} MRR: {}'.format(round(r1,3),round(r2,3),round(mrr,3)))
        return r1, r2, mrr

def calc_test_result_ddrel(pair_ids, predictions, labels, session_ids):
    # calculate session-level prediction
    true_label = labels
    predicted_label = [np.argmax(pred) for pred in predictions]
    session_accuracy = np.sum(np.array(true_label) == np.array(predicted_label)) / len(true_label)

    print("Session-Level Classification")
    print("Confusion Matrix :")
    print(confusion_matrix(true_label, predicted_label))
    print("Classification Report :")        
    print(classification_report(true_label, predicted_label, digits=4))
    p_macro, r_macro, f_macro, support_macro = precision_recall_fscore_support(true_label, predicted_label, average='macro')
    print('Weighted FScore: \n ', p_macro, r_macro, f_macro, support_macro)  
    session_fmacro = f_macro

    # prepare for pair-level prediction
    pairs = {}
    label_cnt = predictions.shape[-1]
    for pair_id, pred, label, session in zip(pair_ids, predictions, labels, session_ids):
        # for each sample, calculate mrr for each rel type:
        logit = sorted([(val, label_idx) for label_idx, val in enumerate(pred)])
        if pair_id not in pairs:
            pairs[pair_id] = {}
            pairs[pair_id]['mrr'] = dict(zip(range(label_cnt),[0]*label_cnt))
            pairs[pair_id]['label'] = label # for each pair, the label is the same
        for rank, (val, label_idx) in enumerate(logit): # record mrr
            pairs[pair_id]['mrr'][label_idx] += 1. / (label_cnt - rank)
    
    # calculate pair-level prediction
    true_label, predicted_label = [], []

    for pair, info in pairs.items():
        label, mrr = info['label'], info['mrr']
        final_pred = max(mrr.items(), key=lambda x: x[1])[0]
        true_label.append(label)
        predicted_label.append(final_pred)
    pair_accuracy = np.sum(np.array(true_label) == np.array(predicted_label)) / len(true_label)


    print("Pair-Level Classification")
    print("Confusion Matrix :")
    print(confusion_matrix(true_label, predicted_label))
    print("Classification Report :")        
    print(classification_report(true_label, predicted_label, digits=4))
    p_macro, r_macro, f_macro, support_macro = precision_recall_fscore_support(true_label, predicted_label, average='macro')
    print('Weighted FScore: \n ', p_macro, r_macro, f_macro, support_macro)
    pair_fmacro = f_macro
    return session_accuracy, session_fmacro, pair_accuracy, pair_fmacro


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)

def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if test_nan and torch.isnan(param_model.grad).sum() > 0:
            is_nan = True
        if param_opti.grad is None:
            param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
        param_opti.grad.data.copy_(param_model.grad.data)
    return is_nan


def f1_eval(logits, labels_all):
    def getpred(result, T1 = 0.5, T2 = 0.4):
        ret = []
        for i in range(len(result)):
            r = []
            maxl, maxj = -1, -1
            for j in range(len(result[i])):
                if result[i][j] > T1:
                    r += [j]
                if result[i][j] > maxl:
                    maxl = result[i][j]
                    maxj = j
            if len(r) == 0:
                if maxl <= T2:
                    r = [36]
                else:
                    r += [maxj]
            ret += [r]
        return ret

    def geteval(devp, data):
        correct_sys, all_sys = 0, 0
        correct_gt = 0
        
        for i in range(len(data)):
            for id in data[i]:
                if id != 36:
                    correct_gt += 1
                    if id in devp[i]:
                        correct_sys += 1

            for id in devp[i]:
                if id != 36:
                    all_sys += 1

        precision = 1 if all_sys == 0 else correct_sys/all_sys
        recall = 0 if correct_gt == 0 else correct_sys/correct_gt
        f_1 = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0
        return f_1

    logits = np.asarray(logits)
    logits = list(1 / (1 + np.exp(-logits)))

    labels = []
    for la in labels_all:
        label = []
        for i in range(36):
            if la[i] == 1:
                label += [i]
        if len(label) == 0:
            label = [36]
        labels += [label]
    assert(len(labels) == len(logits))
    
    bestT2 = bestf_1 = 0
    for T2 in range(51):
        devp = getpred(logits, T2=T2/100.)
        f_1 = geteval(devp, labels)
        if f_1 > bestf_1:
            bestf_1 = f_1
            bestT2 = T2/100.

    return bestf_1, bestT2

def get_logits4eval(model, dataloader, savefile, device, dataset='DialogRE'):
    model.eval()
    logits_all = []
    encoder_attn_all = []
    gtn_weights_all = []
    gtn_adj_all = []
    for batch in tqdm(dataloader, desc="Iteration"):
        input_ids = batch['input_ids'].to(device)
        segment_ids = batch['segment_ids'].to(device)
        input_masks = batch['input_masks'].to(device)
        mention_ids = batch['mention_ids'].to(device)
        speaker_ids = batch['speaker_ids'].to(device)
        label_ids = batch['label_ids'].to(device)
        graphs = batch['graphs']
        # add CLS
        cls_indices = batch['cls_indices'].to(device)
        transcript_ids = batch['transcript_ids']
        answer_ids = batch['answer_ids']
        hidialog_masks = batch['hidialog_masks'].to(device)
        # attention_mask = batch['cls_attn_mask'].to(device)
        with torch.no_grad():
            tmp_eval_loss, logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks, speaker_ids=speaker_ids, graphs=graphs,  labels=label_ids, output_attentions=True, cls_indices=cls_indices, tran_ids=None, hidialog_mask=hidialog_masks)
        
        
        logits = logits.detach().cpu().numpy()
        pred = np.argmax(logits,1)
        for i in range(len(logits)):
            logits_all += [logits[i]]


    with open(savefile, "w") as f:
        for i in range(len(logits_all)):
            for j in range(len(logits_all[i])):
                f.write(str(logits_all[i][j]))
                if j == len(logits_all[i])-1:
                    f.write("\n")
                else:
                    f.write(" ")
    # # attn map related
    # if "logits_test.txt" in savefile:
    #     savefile = re.sub('logits_test','attn_map',savefile)
    #     with open(savefile, 'wb') as f:
    #         pickle.dump([encoder_attn_all,gtn_weights_all,gtn_adj_all],f)

def get_logits4eval_ERC(model, dataloader, savefile, resultsavefile, device, data_name):
    model.eval()
    logits_all = []
    labels_all = []
    sample_ids_all = []
    answer_ids_all = []
    accuracy_all, nb_eval_examples = 0, 0
    for batch in tqdm(dataloader, desc="Iteration"):
        input_ids = batch['input_ids'].to(device)
        segment_ids = batch['segment_ids'].to(device)
        input_masks = batch['input_masks'].to(device)
        mention_ids = batch['mention_ids'].to(device)
        speaker_ids = batch['speaker_ids'].to(device)
        label_ids = batch['label_ids'].to(device)
        graphs = batch['graphs']
        cls_indices = batch['cls_indices'].to(device)
        # attention_mask = batch['cls_attn_mask'].to(device)
        transcript_ids = batch['transcript_ids'] # sample id in MuTual, Pair id in DDRel
        answer_ids = batch['answer_ids'] # answer in MuTual, session id in DDRel
        hidialog_masks = batch['hidialog_masks'].to(device)
        
        with torch.no_grad():
            tmp_eval_loss, logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks, speaker_ids=speaker_ids, graphs=graphs,  labels=label_ids, cls_indices=cls_indices, tran_ids=None, hidialog_mask=hidialog_masks)


        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()   


        for i in range(len(logits)):
            logits_all += [logits[i]]
        for i in range(len(label_ids)):
            labels_all.append(label_ids[i])
        for i in range(len(transcript_ids)):
            sample_ids_all.append(transcript_ids[i])
        for i in range(len(answer_ids)):
            answer_ids_all.append(answer_ids[i])
        if data_name == 'MRDA':
            accuracy_all += accuracy_dac(logits, label_ids)
            nb_eval_examples += input_ids.size(0)

    if data_name == 'MuTual':
        if 'dev' in str(resultsavefile):
            r1, r2, mrr = calc_test_result_mutual(sample_ids_all, logits_all, labels_all)
            with open(resultsavefile, "w") as f:
                f.write("r1 :")
                f.write(str(r1))
                f.write("\n")
                f.write("r2 :")
                f.write(str(r2))
                f.write("\n")
                f.write("mrr :")
                f.write(str(mrr))
                f.write("\n")
        else:
            results = calc_test_result_mutual(sample_ids_all, logits_all, labels_all, answer_ids_all)
            with open(resultsavefile, "w") as f:
                f.writelines(results)
    elif data_name == 'DDRel':
        session_accuracy, session_fmacro, pair_accuracy, pair_fmacro = calc_test_result_ddrel(sample_ids_all, logits_all, labels_all, answer_ids_all)
        with open(resultsavefile, "w") as f:
            f.write("session_accuracy :")
            f.write(str(session_accuracy))
            f.write("\n")
            f.write("session_fmacro :")
            f.write(str(session_fmacro))
            f.write("\n")
            f.write("pair_accuracy :")
            f.write(str(pair_accuracy))
            f.write("\n")    
            f.write("pair_fmacro :")
            f.write(str(pair_fmacro))
            f.write("\n")                        
    else:
        p_weighted, r_weighted, f_weighted, support_weighted = calc_test_result(logits_all, labels_all, data_name)
        
        with open(resultsavefile, "w") as f:
            f.write("p_weighted :")
            f.write(str(p_weighted))
            f.write("\n")
            f.write("r_weighted :")
            f.write(str(r_weighted))
            f.write("\n")
            f.write("f_weighted :")
            f.write(str(f_weighted))
            f.write("\n")
            f.write("support_weighted :")
            f.write(str(support_weighted))
            f.write("\n")
            if data_name == 'MRDA':
                accuracy_all = accuracy_all / nb_eval_examples
                f.write("accuracy_all :")
                f.write(str(accuracy_all))
                f.write("\n")        

    with open(savefile, "w") as f:
        for i in range(len(logits_all)):
            for j in range(len(logits_all[i])):
                f.write(str(logits_all[i][j]))
                if j == len(logits_all[i])-1:
                    f.write("\n")
                else:
                    f.write(" ")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The config json file corresponding to the pre-trained model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--data_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the dataset to train.")
    parser.add_argument("--encoder_type",
                        default=None,
                        type=str,
                        required=True,
                        help="The type of pre-trained model.")
    parser.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary file that the model was trained on.")
    parser.add_argument("--merges_file",
                        default=None,
                        type=str,
                        help="The merges file that the RoBERTa model was trained on.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained model).")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--save_checkpoints_steps",
                        default=1000,
                        type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=666,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=2,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument("--resume",
                        default=False,
                        action='store_true',
                        help="Whether to resume the training.")
    parser.add_argument("--f1eval",
                        default=True,
                        action='store_true',
                        help="Whether to use f1 for dev evaluation during training.")
    parser.add_argument("--gtn_num_layers",
                        type=int,
                        default=2,
                        help="gtn_num_layers.")
    parser.add_argument("--gtn_num_channels",
                        type=int,
                        default=3,
                        help="gtn_num_channels.") 
    parser.add_argument("--node_projection",
                        type=bool,
                        default=False,
                        help="node projection.")     
    parser.add_argument("--layerwise_lr_decay",
                        default=0.9,
                        type=float,
                        help="layerwise_lr_decay.")
    parser.add_argument("--intra_turn_only",
                        default=False,
                        action='store_true',
                        help="whether to use GTN")
                                                


    parser.add_argument("--gpu",type=str,default='0')
    parser.add_argument("--sleep",type=float,default=0.0)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    time.sleep(args.sleep)
    if args.data_name not in n_classes:
        raise ValueError("Data not found: %s" % (args.data_name))
    
    n_class = n_classes[args.data_name]

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
            
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # if n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)
    set_seed(42)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    
    if args.encoder_type == "BERT":
        config = BertConfig.from_json_file(args.config_file)
    elif args.encoder_type == "RoBERTa":
        config = RobertaConfig.from_json_file(args.config_file)
    else:
        raise ValueError("The encoder type is BERT or RoBERTa.")

    if args.max_seq_length > config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
            args.max_seq_length, config.max_position_embeddings))

    if os.path.exists(args.output_dir) and 'model.pt' in os.listdir(args.output_dir):
        if args.do_train and not args.resume:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)
    
    if args.encoder_type == "BERT":
        tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    elif args.encoder_type == "RoBERTa" and args.merges_file:
        tokenizer = RobertaTokenizer(vocab_file=args.vocab_file, merges_file=args.merges_file)
        special_tokens_dict = {'additional_special_tokens': ["[unused1]", "[unused2]"]} #,"[unused3]"
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict) 
    else:
        raise ValueError("The Roberta model needs a merge file.")

    train_set = None
    num_train_steps = None
    if args.do_train:
        train_set = HiDialogDataset(src_file=args.data_dir, save_file=args.data_dir + "/train_" + args.encoder_type + ".pkl", max_seq_length=args.max_seq_length, tokenizer=tokenizer, n_class=n_class, encoder_type=args.encoder_type, dataset=args.data_name)
        num_train_steps = int(
            len(train_set) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        train_loader = HiDialogDataloader(dataset=train_set, batch_size=args.train_batch_size, shuffle=True, relation_num=n_class, max_length=args.max_seq_length,dataset_name=args.data_name)
    
    # exit()

    # add deberta
    if args.encoder_type == "BERT":
        model = HiDialog_BERT(config, n_class)
        if args.init_checkpoint is not None:
            model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'), strict=False)
    elif args.encoder_type == "RoBERTa":
        model = HiDialog_RoBERTa(config, n_class, gtn_num_layers=args.gtn_num_layers, gtn_num_channels=args.gtn_num_channels, dataset=args.data_name, intra_turn_only=args.intra_turn_only)
        if args.init_checkpoint is not None:
            model.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'), strict=False)
        model.roberta.resize_token_embeddings(len(tokenizer))

    if args.fp16:
        model.half()
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                            for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'gamma', 'beta']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.0},
    #     {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}
    #     ]
    
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}
            ]

    optimizer = BERTAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    global_step = 0

    if args.resume:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt")))

    if args.do_eval:
        dev_set = HiDialogDataset(src_file=args.data_dir, save_file=args.data_dir + "/dev_" + args.encoder_type + ".pkl", max_seq_length=args.max_seq_length, tokenizer=tokenizer, n_class=n_class, encoder_type=args.encoder_type,dataset=args.data_name)
        dev_loader = HiDialogDataloader(dataset=dev_set, batch_size=args.eval_batch_size, shuffle=False, relation_num=n_class, max_length=args.max_seq_length, dataset_name=args.data_name)
        test_set = HiDialogDataset(src_file=args.data_dir, save_file=args.data_dir + "/test_" + args.encoder_type + ".pkl", max_seq_length=args.max_seq_length, tokenizer=tokenizer, n_class=n_class, encoder_type=args.encoder_type,dataset=args.data_name)
        test_loader = HiDialogDataloader(dataset=test_set, batch_size=args.eval_batch_size, shuffle=False, relation_num=n_class, max_length=args.max_seq_length,dataset_name=args.data_name)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    # assert 1 == 2

    if args.do_train:
        best_metric = 0
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_set))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        for ep in trange(int(args.num_train_epochs), desc="Epoch"):
            
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
                input_ids = batch['input_ids'].to(device)
                segment_ids = batch['segment_ids'].to(device)
                input_masks = batch['input_masks'].to(device)
                mention_ids = batch['mention_ids'].to(device)
                speaker_ids = batch['speaker_ids'].to(device)
                label_ids = batch['label_ids'].to(device)
                graphs = batch['graphs']
                cls_indices = batch['cls_indices'].to(device) # add CLS
                # attention_mask = batch['cls_attn_mask'].to(device)
                # print((input_masks==1).sum(1))
                transcript_ids = batch['transcript_ids']
                answer_ids = batch['answer_ids']
                hidialog_masks = batch['hidialog_masks'].to(device)

                loss, _ = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks, speaker_ids=speaker_ids, graphs=graphs,  labels=label_ids, cls_indices=cls_indices, tran_ids=None, hidialog_mask=hidialog_masks)
                if n_gpu > 1:
                    loss = loss.mean()
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:

                        optimizer.step()

                    model.zero_grad()
                    global_step += 1
            
            
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            logits_all = []
            labels_all = []
            ids_all = []
            answer_ids_all = []
            for batch in dev_loader:
                input_ids = batch['input_ids'].to(device)
                segment_ids = batch['segment_ids'].to(device)
                input_masks = batch['input_masks'].to(device)
                mention_ids = batch['mention_ids'].to(device)
                speaker_ids = batch['speaker_ids'].to(device)
                label_ids = batch['label_ids'].to(device)
                graphs = batch['graphs']
                cls_indices = batch['cls_indices'].to(device) # add CLS
                # attention_mask = batch['cls_attn_mask'].to(device)
                transcript_ids = batch['transcript_ids']
                answer_ids = batch['answer_ids']
                hidialog_masks = batch['hidialog_masks'].to(device)
                with torch.no_grad():
                    tmp_eval_loss, logits  = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks, speaker_ids=speaker_ids, graphs=graphs,  labels=label_ids, cls_indices=cls_indices, tran_ids=None, hidialog_mask=hidialog_masks)
                
                logits_copy = copy.deepcopy(logits) 
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                for i in range(len(logits)):
                    logits_all += [logits[i]]
                for i in range(len(label_ids)):
                    labels_all.append(label_ids[i])
                for i in range(len(transcript_ids)):
                    ids_all.append(transcript_ids[i])
                for i in range(len(answer_ids)):
                    answer_ids_all.append(answer_ids[i])

                if args.data_name == "DialogRE":
                    tmp_eval_accuracy = accuracy(logits, label_ids.reshape(-1))

                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1
                elif args.data_name in ["MRDA"]:
                    tmp_eval_accuracy = accuracy_dac(logits, label_ids)

                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1
                else:
                    eval_loss += tmp_eval_loss.mean().item()

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

            if args.data_name == "DialogRE":
                eval_loss = eval_loss / nb_eval_steps
                eval_accuracy = eval_accuracy / nb_eval_examples

                if args.do_train:
                    result = {'eval_loss': eval_loss,
                            'global_step': global_step,
                            'loss': tr_loss/nb_tr_steps}
                else:
                    result = {'eval_loss': eval_loss}

                if args.f1eval:
                    eval_f1, eval_T2 = f1_eval(logits_all, labels_all)
                    result["f1"] = eval_f1
                    result["T2"] = eval_T2                

                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))

                if args.f1eval:
                    if eval_f1 >= best_metric:
                        torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pt"))
                        best_metric = eval_f1
                else:
                    if eval_accuracy >= best_metric:
                        torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pt"))
                        best_metric = eval_accuracy              
            elif args.data_name in ["MRDA"]:
                eval_loss = eval_loss / nb_eval_steps
                eval_accuracy = eval_accuracy / nb_eval_examples
                if args.do_train:
                    result = {'eval_loss': eval_loss,
                            'eval_accuracy': eval_accuracy,
                            'global_step': global_step,
                            'loss': tr_loss/nb_tr_steps}
                else:
                    result = {'eval_loss': eval_loss,
                            'eval_accuracy': eval_accuracy}
                
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                
                if eval_accuracy >= best_metric:
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pt"))
                    best_metric = eval_accuracy
                torch.save(model.state_dict(), os.path.join(args.output_dir, "model_{}.pt".format(ep)))                
                
            else:
                # nb_tr_steps = 1
                eval_loss = eval_loss / nb_eval_steps

                if args.do_train:
                    result = {'eval_loss': eval_loss,
                            'global_step': global_step,
                            'loss': tr_loss/nb_tr_steps}
                else:
                    result = {'eval_loss': eval_loss}

                if args.f1eval:
                    p_weighted, r_weighted, eval_f1, support_weighted = calc_test_result(logits_all, labels_all, args.data_name)
                    result["f1"] = eval_f1      
                    if eval_f1 >= best_metric:
                        torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pt"))
                        best_metric = eval_f1   
                            
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "model_{}.pt".format(ep)))
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))

        model.load_state_dict(torch.load(os.path.join(args.output_dir, "model_best.pt")))
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))

    model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt")))

    if args.do_eval:
        if args.data_name == "DialogRE":
            # for f1
            dev_set = HiDialogDataset(src_file=args.data_dir, save_file=args.data_dir + "/dev_" + args.encoder_type + ".pkl", max_seq_length=args.max_seq_length, tokenizer=tokenizer, n_class=n_class, encoder_type=args.encoder_type)
            dev_loader = HiDialogDataloader(dataset=dev_set, batch_size=args.eval_batch_size, shuffle=False, relation_num=n_class, max_length=args.max_seq_length)
            test_set = HiDialogDataset(src_file=args.data_dir, save_file=args.data_dir + "/test_" + args.encoder_type + ".pkl", max_seq_length=args.max_seq_length, tokenizer=tokenizer, n_class=n_class, encoder_type=args.encoder_type)
            test_loader = HiDialogDataloader(dataset=test_set, batch_size=args.eval_batch_size, shuffle=False, relation_num=n_class, max_length=args.max_seq_length)

            #for f1c
            devc_set = HiDialogDataset4f1c(src_file=args.data_dir, save_file=args.data_dir + "/devc_" + args.encoder_type + ".pkl", max_seq_length=args.max_seq_length, tokenizer=tokenizer, n_class=n_class, encoder_type=args.encoder_type)
            devc_loader = HiDialogDataloader(dataset=devc_set, batch_size=args.eval_batch_size, shuffle=False, relation_num=n_class, max_length=args.max_seq_length)
            testc_set = HiDialogDataset4f1c(src_file=args.data_dir, save_file=args.data_dir + "/testc_" + args.encoder_type + ".pkl", max_seq_length=args.max_seq_length, tokenizer=tokenizer, n_class=n_class, encoder_type=args.encoder_type)
            testc_loader = HiDialogDataloader(dataset=testc_set, batch_size=args.eval_batch_size, shuffle=False, relation_num=n_class, max_length=args.max_seq_length)

            get_logits4eval(model, dev_loader, os.path.join(args.output_dir, "logits_dev.txt"), device)
            get_logits4eval(model, test_loader, os.path.join(args.output_dir, "logits_test.txt"), device)
            get_logits4eval(model, devc_loader, os.path.join(args.output_dir, "logits_devc.txt"), device)
            get_logits4eval(model, testc_loader, os.path.join(args.output_dir, "logits_testc.txt"), device)
        else:
            # for f1
            dev_set = HiDialogDataset(src_file=args.data_dir, save_file=args.data_dir + "/dev_" + args.encoder_type + ".pkl", max_seq_length=args.max_seq_length, tokenizer=tokenizer, n_class=n_class, encoder_type=args.encoder_type,dataset=args.data_name)
            dev_loader = HiDialogDataloader(dataset=dev_set, batch_size=args.eval_batch_size, shuffle=False, relation_num=n_class, max_length=args.max_seq_length,dataset_name=args.data_name)
            test_set = HiDialogDataset(src_file=args.data_dir, save_file=args.data_dir + "/test_" + args.encoder_type + ".pkl", max_seq_length=args.max_seq_length, tokenizer=tokenizer, n_class=n_class, encoder_type=args.encoder_type,dataset=args.data_name)
            test_loader = HiDialogDataloader(dataset=test_set, batch_size=args.eval_batch_size, shuffle=False, relation_num=n_class, max_length=args.max_seq_length,dataset_name=args.data_name)

            get_logits4eval_ERC(model, dev_loader, os.path.join(args.output_dir, "logits_dev.txt"), os.path.join(args.output_dir, "dev_result.txt"), device, args.data_name)
            get_logits4eval_ERC(model, test_loader, os.path.join(args.output_dir, "logits_test.txt"), os.path.join(args.output_dir, "test_result.txt"), device, args.data_name)

if __name__ == "__main__":
    main()
