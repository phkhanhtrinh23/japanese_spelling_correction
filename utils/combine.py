import argparse
import os
import re
import unicodedata
import json
from collections import Counter
from multiprocessing import Pool

from errorify import Errorify
from edits import EditTagger
from helpers import write_dataset


en_sentence_re = re.compile(r'([a-zA-Z]+[\W]*\s]){5,}')
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)


def preprocess(source_dir, output_dir):
    if not os.path.isdir(source_dir):
        raise ValueError(f'address text folder not found at {source_dir}')

    ja_file = open("data/corpora/ja_input.txt", "w", encoding="utf-8")

    for root, dirs, files in os.walk(source_dir):
        if not dirs:
            direc = root.split("/")[-1]
            pre_direc = root.split("/")[-2]
            if direc == "jpn-eng":
                for filename in files:
                    f = open(os.path.join(root, filename), encoding="utf-8")
                    lines = f.readlines()
                    print("direc: {}, length: {}".format(direc, len(lines)))
                    for line in lines:
                        _, line, _ = line.split("\t")
                        line = line.strip()
                        line = emoji_pattern.sub(r'', line)
                        if not line or line[0] == '<':
                            continue
                        if en_sentence_re.search(line):
                            continue
                        ja_file.write(line + "\n")
            elif direc == "jp-address":
                for filename in files:
                    f = open(os.path.join(root, filename), encoding="utf-8")
                    lines = f.readlines()
                    print("direc: {}, length: {}".format(direc, len(lines)))
                    for line in lines:
                        line = line.strip()
                        line = emoji_pattern.sub(r'', line)
                        if not line or line[0] == '<':
                            continue
                        if en_sentence_re.search(line):
                            continue
                        ja_file.write(line + "\n")
            elif direc == "BSD":
                for filename in files:
                    if filename.split(".")[-1] == "json":
                        f = open(os.path.join(root, filename), encoding="utf-8")
                        data_json = json.load(f)
                        lines = []
                        for i, info in enumerate(data_json):
                            for content in info["conversation"]:
                                lines.append(content["ja_speaker"])
                                lines.append(content["ja_sentence"])
                        print("direc: {}, length: {}".format(direc, len(lines)))
                        for line in lines:
                            line = line.strip()
                            line = emoji_pattern.sub(r'', line)
                            if not line or line[0] == '<':
                                continue
                            if en_sentence_re.search(line):
                                continue
                            ja_file.write(line + "\n")
            elif pre_direc == "PheMT" and direc in ["abbrev", "colloq", "proper", "variant"]:
                for filename in files:
                    if len(filename.split('.')) > 2 and filename.split('.')[-2] == 'norm':
                        f = open(os.path.join(root, filename), encoding="utf-8")
                        lines = f.readlines()
                    elif len(filename.split('.')) == 2 and filename.split('.')[-1] == 'ja':
                        f = open(os.path.join(root, filename), encoding="utf-8")
                        lines = f.readlines()
                    else:
                        continue
                    print("direc: {}, length: {}".format(direc, len(lines)))
                    for line in lines:
                        line = line.strip()
                        line = emoji_pattern.sub(r'', line)
                        if not line or line[0] == '<':
                            continue
                        if en_sentence_re.search(line):
                            continue
                        ja_file.write(line + "\n")
    ja_file.close()


def main(args):
    preprocess(args.source_dir, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_dir',
                        help='Path to text folder extracted by address',
                        default="./data/corpora/")
    parser.add_argument('-o', '--output_dir',
                        help='Path to output directory',
                        default="./data/saved_data")
    args = parser.parse_args()
    main(args)
