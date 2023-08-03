import argparse
import os
import json

import tensorflow as tf
from tensorflow import keras
import numpy as np
from transformers import AdamWeightDecay
from sklearn.metrics import classification_report
from utils.preprocess import preprocess_file

from model import GEC
from utils.helpers import read_dataset, WeightedSCCE


AUTO = tf.data.AUTOTUNE


def train(corpora_dir, output_weights_path, vocab_dir, transforms_file,
          pretrained_weights_path, batch_size, n_epochs, dev_ratio, dataset_len,
          dataset_ratio, bert_trainable, learning_rate, class_weight_path,
          source_file, output_dir, use_existing, processes,
          filename='edit_tagged_sentences.tfrec.gz'):
    print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))

    files = [os.path.join(root, filename)
            for root, dirs, files in tf.io.gfile.walk(corpora_dir)
            if filename in files]
    dataset = read_dataset(files).shuffle(buffer_size=1024)
    
    # Get dataset_len
    dataset_len = [i for i,_ in enumerate(dataset)][-1] + 1
    if dataset_len:
        dataset_card = tf.data.experimental.assert_cardinality(dataset_len)
        dataset = dataset.apply(dataset_card)
    if 0 < dataset_ratio < 1:
        dataset_len = int(dataset_len * dataset_ratio)
        dataset = dataset.take(dataset_len)
    
    print('Length:', dataset_len, tf.data.experimental.cardinality(dataset))
    print('Loaded dataset:', dataset, dataset.cardinality().numpy())

    dev_len = int(dataset_len * dev_ratio)
    train_set = dataset.skip(dev_len).prefetch(AUTO)
    dev_set = dataset.take(dev_len).prefetch(AUTO)
    
    print('Ratio of train and dev set:', train_set.cardinality().numpy(), dev_set.cardinality().numpy())
    print(f'Using {dev_ratio} of dataset for dev set')
    
    train_set = train_set.batch(batch_size, num_parallel_calls=AUTO)
    dev_set = dev_set.batch(batch_size, num_parallel_calls=AUTO)

    gec = GEC(vocab_path=vocab_dir, verb_adj_forms_path=transforms_file,
        pretrained_weights_path=pretrained_weights_path,
        bert_trainable=bert_trainable, learning_rate=learning_rate)
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=output_weights_path + '_checkpoint',
        save_weights_only=True,
        monitor='val_labels_probs_accuracy',
        mode='max',
        save_best_only=True)
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='loss', patience=3)
    gec.model.fit(train_set, epochs=n_epochs, validation_data=dev_set,
        callbacks=[model_checkpoint_callback, early_stopping_callback])
    gec.model.save_weights(output_weights_path)

        # # Generate data at every epoch
        # print("Generating new data set at the end of epoch: {}...".format(epoch_i))
        # preprocess_file(source_file, output_dir, processes, use_existing)
        # print("Finished generating data.")

def main(args):
    train(args.corpora_dir, args.output_weights_path, args.vocab_dir,
          args.transforms_file, args.pretrained_weights_path, args.batch_size,
          args.n_epochs, args.dev_ratio, args.dataset_len, args.dataset_ratio,
          args.bert_trainable, args.learning_rate, args.class_weight_path,
          args.source_file, args.output_dir, args.use_existing, args.processes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpora_dir',
                        help='Path to dataset folder',
                        default="./utils/data/saved_data/ja_input")
    parser.add_argument('-o', '--output_weights_path',
                        help='Path to save model weights to',
                        default="./weights/checkpoints")
    parser.add_argument('-v', '--vocab_dir',
                        help='Path to output vocab folder',
                        default='./utils/data/output_vocab')
    parser.add_argument('-t', '--transforms_file',
                        help='Path to verb/adj transforms file',
                        default='./utils/data/transform.txt')
    parser.add_argument('-p', '--pretrained_weights_path',
                        help='Path to pretrained model weights',
                        default='./utils/data/model/model_checkpoint')
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Number of samples per batch',
                        default=256)
    parser.add_argument('-e', '--n_epochs', type=int,
                        help='Number of epochs',
                        default=5)
    parser.add_argument('-d', '--dev_ratio', type=float,
                        help='Percent of whole dataset to use for dev set',
                        default=0.2)
    parser.add_argument('-l', '--dataset_len', type=int,
                        help='Cardinality of dataset')
    parser.add_argument('-r', '--dataset_ratio', type=float,
                        help='Percent of whole dataset to use',
                        default=1.0)
    parser.add_argument('-bt', '--bert_trainable',
                        help='Enable training for BERT encoder layers',
                        action='store_true')
    parser.add_argument('-lr', '--learning_rate', type=float,
                        help='Learning rate',
                        default=1e-4)
    parser.add_argument('-cw', '--class_weight_path',
                        help='Path to class weight file')

    parser.add_argument('-s', '--source_file',
                        help='Path to text folder extracted by address',
                        default="./utils/data/corpora/ja_input.txt")
    parser.add_argument('-ou', '--output_dir',
                        help='Path to output directory',
                        default="./utils/data/saved_data")
    parser.add_argument('-pr', '--processes', type=int,
                        help='Number of processes',
                        required=False)
    parser.add_argument('-ue', '--use_existing',
                        help='Edit tag existing error-generated sentences',
                        type=bool,
                        default=False)
    
    args = parser.parse_args()
    main(args)
