import argparse
import os
import re
import unicodedata
import json
from collections import Counter
from multiprocessing import Pool

from .errorify import Errorify
from .edits import EditTagger
from .helpers import write_dataset


en_sentence_re = re.compile(r'([a-zA-Z]+[\W]*\s]){5,}')
# ja_sentence_re = re.compile(u'([\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uff9f\u4e00-\u9faf\u3400-\u4dbf])', re.U)
# ja_sentence_re = re.compile(r'([ぁ-んァ-ン一-龥])')
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

errorify = Errorify()
edit_tagger = EditTagger()


def preprocess_file_part(args,
                        correct_file='corr_sentences.txt',
                        incorrect_file='incorr_sentences.txt',
                        edit_tags_file='edit_tagged_sentences.tfrec.gz'):
    edit_tagger.edit_freq = Counter()
    fp, output_dir, use_existing = args
    edit_rows = []
    fp 
    base_path = os.path.join(output_dir, "ja_input")
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    corr_path = os.path.join(base_path, correct_file)
    incorr_path = os.path.join(base_path, incorrect_file)
    if use_existing:
        with open(corr_path, 'r', encoding='utf-8') as f:
            corr_lines = f.readlines()
        with open(incorr_path, 'r', encoding='utf-8') as f:
            incorr_lines = f.readlines()
    else:
        corr_lines = []
        incorr_lines = []
        with open(fp, encoding='utf-8') as file:
            lines = file.readlines()
            print("length: {}, file: {}".format(len(lines), file))
            for line in lines:
                line = line.strip()
                line = emoji_pattern.sub(r'', line)
                if not line or line[0] == '<':
                    continue
                if en_sentence_re.search(line):
                    continue
                line = unicodedata.normalize('NFKC', line).replace(' ', '')
                quote_lvl = 0
                brackets_lvl = 0
                start_i = 0
                sents = []
                for i, c in enumerate(line):
                    if c == '「':
                        quote_lvl += 1
                    elif c == '」':
                        quote_lvl -= 1
                    elif c == '(':
                        brackets_lvl += 1
                    elif c == ')':
                        brackets_lvl -= 1
                    elif i == (len(line) - 1) and quote_lvl == 0 and brackets_lvl == 0:
                        sents.append(line)
                for sent in sents:
                    sent = sent.strip().lstrip('。')
                    if not sent:
                        continue
                    error_sent = errorify(sent)
                    error_sent = error_sent.strip().lstrip('。')
                    if len(sent) == 0 or len(error_sent) == 0:
                        continue
                    else:
                        corr_lines.append(f'{sent}\n')
                        incorr_lines.append(f'{error_sent}\n')
        with open(corr_path, 'w', encoding='utf-8') as file:
            file.writelines(corr_lines)
        with open(incorr_path, 'w', encoding='utf-8') as file:
            file.writelines(incorr_lines)
    for incorr_line, corr_line in zip(incorr_lines, corr_lines):
        incorr_line = incorr_line.strip()
        corr_line = corr_line.strip()
        if not incorr_line or not corr_line:
            continue
        levels = edit_tagger(incorr_line, corr_line)
        edit_rows.extend(levels)
    edit_tags_path = os.path.join(base_path, edit_tags_file)
    write_dataset(edit_tags_path, edit_rows)
    print(f'Processed {len(corr_lines)} sentences, ' \
          f'{len(edit_rows)} edit-tagged sentences in {fp}')
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
          'sentences from ja_input.txt')
    print(f'Synthetic error corpus output to {output_dir}')


def main(args):
    preprocess_file(args.source_file, args.output_dir, args.processes,
                args.use_existing)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_file',
                        help='Path to text folder extracted by address',
                        default="./data/corpora/ja_input.txt")
    parser.add_argument('-o', '--output_dir',
                        help='Path to output directory',
                        default="./data/saved_data")
    parser.add_argument('-p', '--processes', type=int,
                        help='Number of processes',
                        required=False)
    parser.add_argument('-e', '--use_existing',
                        help='Edit tag existing error-generated sentences',
                        type=bool,
                        default=False)
    args = parser.parse_args()
    main(args)
