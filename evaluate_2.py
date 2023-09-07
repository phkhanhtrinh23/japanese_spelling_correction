import os
import argparse

import tensorflow as tf
from fugashi import Tagger
from nltk.translate.gleu_score import corpus_gleu

from model import GEC
import pandas as pd
from unidecode import unidecode
import unicodedata
import time
import difflib

tagger = Tagger('-Owakati')

class Solution(object):  
    def findTheDifference(self, s, t):  
        ls_s = [s[i] for i in range(len(s))]  
        ls_t = [t[i] for i in range(len(t))]  
        for elem in ls_s:
            if elem in ls_t:
                ls_t.remove(elem)  
        if len(ls_t) > 0:
            return(len(ls_t))
        else:
            return 0
obj = Solution()

def tokenize(sentence):
    return [t.surface for t in tagger(sentence)]


def main(weights_path, vocab_dir, transforms_file, corpus_dir):
    df = pd.read_csv(os.path.join(corpus_dir, 'DLIC-Add.csv'))
    source_sents = [line for line in df["Address-predict"]]
    reference_sents = [line for line in df["Address-label"]]

    temp_src, temp_lab = [], []
    new_source_sents, new_reference_sents = [], []
    for src_sent, lab_sent in zip(source_sents, reference_sents):
        if isinstance(src_sent, str):
            converted_sentence = unicodedata.normalize('NFKC', src_sent).replace(' ', '')
            new_source_sents.append(converted_sentence)
            
            converted_sentence = unicodedata.normalize('NFKC', lab_sent).replace(' ', '')
            new_reference_sents.append(converted_sentence)
    source_sents = new_source_sents
    reference_sents = new_reference_sents

    print(f'Loaded {len(source_sents)} src, {len(reference_sents)} ref')

    gec = GEC(vocab_path=vocab_dir, verb_adj_forms_path=transforms_file,
              pretrained_weights_path=weights_path)
    
    pred_tokens, pred_tokens_2 = [], []
    source_batches = [source_sents[i:i + 64]
                      for i in range(0, len(source_sents), 64)]
    temp_time = []
    for i, source_batch in enumerate(source_batches):
        print(f'Predict batch {i+1}/{len(source_batches)}')
        pred_batch = gec.correct(source_batch)
        print("batch", source_batch[0])
        start = time.time()
        _ = gec.correct([source_batch[0]])
        end = time.time()
        temp_time.append(end-start)
        print("avg time {}: {}".format(i, end-start))
        pred_batch_tokens = [sent for sent in pred_batch]
        pred_tokens.extend(pred_batch_tokens)
    
    print("total avg time: {}".format(sum(temp_time)/len(temp_time)))
    
    status = []
    total_length, ratio_list, sent_ref, pred_ref, pred_length,sent_length = [], [], [], [], [], []
    ratio_list_sent = []
    wrong_cases_pred, wrong_cases_sent = [], []
    
    for sent, ref, pred in zip(new_source_sents, reference_sents, pred_tokens):
        seq=difflib.SequenceMatcher(a=pred.lower(), b=ref.lower())
        d=seq.ratio()*100
        
        s_r=obj.findTheDifference(sent, ref)
        p_r=obj.findTheDifference(pred, ref)
        
        r_s=obj.findTheDifference(ref, sent)
        r_p=obj.findTheDifference(ref, pred)
        
        ratio_r_s=r_s/len(ref)*100
        ratio_r_p=r_p/len(ref)*100
        
        print("Ratio:", d)
        print("Ratio ref sent:", ratio_r_s)
        print("Ratio ref pred:", ratio_r_p)
        print("sent - ref:", s_r)
        print("pred - ref:", p_r)
        print("ref - sent:", r_s)
        print("ref - pred:", r_p)
        
        seq=difflib.SequenceMatcher(a=sent.lower(), b=ref.lower())
        d_sent=seq.ratio()*100
        
        total_length.append(len(ref))
        pred_length.append(len(pred))
        sent_length.append(len(sent))
        
        ratio_list_sent.append(d_sent)
        ratio_list.append(d)
        sent_ref.append(s_r)
        pred_ref.append(p_r)
        
        wrong_cases_sent.append(ratio_r_s)
        wrong_cases_pred.append(ratio_r_p)
        
        if sent != ref and pred != ref: # incor -> incor
            status.append(0)
            print("SEN:", sent)
            print("PRE:", pred)
            print("REF:", ref)
            print()
        elif sent != ref and pred == ref: # incor -> cor *
            status.append(1)
            print("PRE-COR:", pred)
            print("REF-COR:", ref)
            print()
        elif sent == ref and pred != ref: # cor -> incor
            status.append(2)
        elif sent == ref and pred == ref: # cor -> cor *
            status.append(3)
            
    print("incor->incor: {}%\nincor->cor: {}%\ncor->incor: {}%\ncor->cor: {}%".format(status.count(0)/len(status)*100,\
                                                                                     status.count(1)/len(status)*100,\
                                                                                     status.count(2)/len(status)*100,\
                                                                                     status.count(3)/len(status)*100))
    print("total length of references' characters:", sum(total_length))
    print("total length of predictions' characters:", sum(pred_length))
    print("total length of inputs' characters:", sum(sent_length))
    print("avg ratio of errors of missing sent:", 100-(sum(ratio_list)/len(ratio_list)))
    print("avg ratio of errors of missing pred:", 100-(sum(ratio_list_sent)/len(ratio_list_sent)))
    print("avg ratio of errors of wrong sent:", sum(wrong_cases_pred)/len(wrong_cases_pred))
    print("avg ratio of errors of wrong pred:", sum(wrong_cases_sent)/len(wrong_cases_sent))
    print("total difference sent - ref characters:", sum(sent_ref))
    print("total difference pred - ref characters:", sum(pred_ref))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights',
                        help='Path to model weights',
                        default='./weights/checkpoint_gectorjsc')
    parser.add_argument('-v', '--vocab_dir',
                        help='Path to output vocab folder',
                        default='./utils/data/output_vocab')
    parser.add_argument('-f', '--transforms_file',
                        help='Path to verb/adj transforms file',
                        default='./utils/data/transform.txt')
    parser.add_argument('-c', '--corpus_dir',
                        help='Path to directory of TMU evaluation corpus',
                        default='./utils/data/corpora')
    args = parser.parse_args()
    main(args.weights, args.vocab_dir, args.transforms_file, args.corpus_dir)
    