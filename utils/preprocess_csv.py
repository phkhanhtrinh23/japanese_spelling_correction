import argparse
import os
import re
import unicodedata
import json
from collections import Counter
from multiprocessing import Pool
from difflib import SequenceMatcher
from edits import EditTagger
from helpers import write_dataset

import pandas as pd


en_sentence_re = re.compile(r'([a-zA-Z]+[\W]*\s]){5,}')
# ja_sentence_re = re.compile(u'([\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uff9f\u4e00-\u9faf\u3400-\u4dbf])', re.U)
# ja_sentence_re = re.compile(r'([ぁ-んァ-ン一-龥])')
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
ja_re = re.compile(r'([ぁ-んァ-ン])')
html_re = re.compile(r'<(\/?[a-z]+)>')
edit_tagger = EditTagger()
sline_re = re.compile(r'\[sline\].*?\[/sline\]')
color_tags = ['[f-blue]','[/f-blue]',
              '[f-red]','[/f-red]',
              '[f-bold]','[/f-bold]']

def clean_line(line):
    line = unicodedata.normalize('NFKC', line.strip()).replace(' ', '')
    # if line.endswith('GOOD'):
    #     line = line[:-4]
    # elif line.endswith('OK'):
    #     line = line[:-2]
    for tag in color_tags:
        line = line.replace(tag, '')
    # line = sline_re.sub('', line).replace('[/sline]', '')
    return line

edit_tagger = EditTagger()


def preprocess_file_part(args,
                        correct_file='corr_sentences.txt',
                        incorrect_file='incorr_sentences.txt',
                        edit_tags_file='edit_tagged_sentences.tfrec.gz'):
    edit_tagger.edit_freq = Counter()
    fp, output_dir, use_existing = args
    edit_rows = []
    fp 
    base_path = os.path.join(output_dir, "compare_add_jp")
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    corr_path = os.path.join(base_path, correct_file)
    incorr_path = os.path.join(base_path, incorrect_file)
    edit_tags_path = os.path.join(base_path, edit_tags_file)
    if use_existing:
        with open(corr_path, 'r', encoding='utf-8') as f:
            corr_lines = f.readlines()
        with open(incorr_path, 'r', encoding='utf-8') as f:
            incorr_lines = f.readlines()
    else:
        corr_lines = []
        incorr_lines = []
        df = pd.read_csv(fp)
        ocr_lines = [line.strip() for line in df["ocr_address_1"]] + [line.strip() for line in df["ocr_address_2"]]
        label_lines = [line.strip() for line in df["label_address_1"]] + [line.strip() for line in df["label_address_2"]]
        pairs = set()
        for learner_sent, target_sent in zip(ocr_lines, label_lines):
            # if not ja_re.search(learner_sent) or html_re.search(learner_sent):
            #     print("Error ja:", learner_sent)
            #     print()
            #     continue
            learner_sent = clean_line(learner_sent)
            target_sent = clean_line(target_sent)
            pairs.add((learner_sent, target_sent))
            
            print("Not error ja input:", learner_sent)
            print("Not error ja target:", target_sent)
            print()
        corr_lines = []
        incorr_lines = []
        edit_rows = []
        for learner_sent, target_sent in pairs:
            # remove appended comments
            matcher = SequenceMatcher(None, learner_sent, target_sent)
            diffs = list(matcher.get_opcodes())
            tag, i1, i2, j1, j2 = diffs[-1]
            if tag == 'insert' and (learner_sent[-1] in '。.!?' or j2 - j1 >= 10):
                target_sent = target_sent[:j1]
            elif tag == 'replace' and (j2 - j1) / (i2 - i1) >= 10:
                continue
            corr_lines.append(f'{target_sent}\n')
            incorr_lines.append(f'{learner_sent}\n')
            levels = edit_tagger(learner_sent, target_sent, levels=True)
            edit_rows.extend(levels)
    with open(corr_path, 'w', encoding='utf-8') as f:
        f.writelines(corr_lines)
    with open(incorr_path, 'w', encoding='utf-8') as f:
        f.writelines(incorr_lines)
    write_dataset(edit_tags_path, edit_rows)
    print(f'Processed {len(corr_lines)} sentences, ' \
        f'{len(edit_rows)} edit-tagged sentences to {base_path}')
    return len(corr_lines), len(edit_rows), edit_tagger.edit_freq


def preprocess_file(source_file, output_dir, processes, use_existing,
                    edit_freq_file='edit_freq.json'):
    """Generate synthetic error corpus."""
    n_sents = 0
    n_edit_sents = 0
    pool_inputs = []
    pool_inputs.append([source_file, output_dir, use_existing])
    print("pool_inputs:", pool_inputs)
    print(f'Processing {len(pool_inputs)} parts...')
    pool = Pool(processes)
    pool_outputs = pool.imap_unordered(preprocess_file_part, pool_inputs)
    n_sents = 0
    n_edit_sents = 0
    edit_freq = Counter()
    for n in pool_outputs:
        n_sents += n[0]
        n_edit_sents += n[1]
        edit_freq.update(n[2])

    with open(os.path.join(output_dir, edit_freq_file), 'w') as f:
        json.dump(edit_freq, f)
    print(f'Processed {n_sents} sentences, {n_edit_sents} edit-tagged ' \
          'sentences from compare_add_jp.csv')
    print(f'Synthetic error corpus output to {output_dir}')


def main(args):
    preprocess_file(args.source_file, args.output_dir, args.processes,
                args.use_existing)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_file',
                        help='Path to text folder extracted by address',
                        default="./utils/data/corpora/compare_add_jp.csv")
    parser.add_argument('-o', '--output_dir',
                        help='Path to output directory',
                        default="./utils/data/saved_data")
    parser.add_argument('-p', '--processes', type=int,
                        help='Number of processes',
                        required=False)
    parser.add_argument('-e', '--use_existing',
                        help='Edit tag existing error-generated sentences',
                        type=bool,
                        default=False)
    args = parser.parse_args()
    main(args)
