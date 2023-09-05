# Grammatical Error Correction for Japanese Spelling Correction - GECToR JSC

GECToR-JSC is described in the paper [GECToR -Grammatical Error Correction: Tag, Not Rewrite](https://arxiv.org/abs/2005.12592), but it is implemented for Japanese. This project's code is based on the official implementation [gector](https://github.com/grammarly/gector).

The [bert-base-japanese](https://huggingface.co/cl-tohoku/bert-base-japanese-v2) used in this project was provided by Tohoku University NLP Lab.

## Datasets

- [Japanese Wikipedia dump](https://dumps.wikimedia.org/), extracted with [WikiExtractor](https://github.com/attardi/wikiextractor), synthetic errors generated using preprocessing scripts
  - 19,841,767 training sentences
- [NAIST Lang8 Learner Corpora](https://sites.google.com/site/naistlang8corpora/)
  - 6,066,306 training sentences (generated from 3,084,0376 original sentences)
- [PheMT](https://github.com/cl-tohoku/PheMT), extracted from this [paper](https://arxiv.org/pdf/2011.02121.pdf)
- [BSD](https://github.com/tsuruoka-lab/BSD), extracted from this [paper](https://arxiv.org/pdf/2008.01940.pdf)
- [jpn-eng](http://www.manythings.org/anki/)
- [jpn-address](https://drive.google.com/drive/folders/1kBz8wbYztRkgz2nQgQvBD1wkWz8Jwz1-?usp=sharing)

### Synthetically Generated Error Corpus

The **JaWiki**, **Lang8**, **BSD**, **PheMT**, **jpn-eng**, and **jp_address** are to synthetically generate errorful sentences, with a method similar to [Awasthi et al. 2019](https://github.com/awasthiabhijeet/PIE/tree/master/errorify), but with adjustments for Japanese. The details of the implementation can be found in the [preprocessing scripts](https://github.com/phkhanhtrinh23/gector_jsc/blob/main/utils/preprocess.py) in this repository.

Example error-generated sentence:
```
西口側には宿泊施設や地元の日本酒や海、山の幸を揃えた飲食店、呑み屋など多くある。        # Correct
西口側までは宿泊から施設や地元の日本酒や、山の幸を揃えた飲食は店、呑み屋など多くあろう。 # Errorful
```

## Model Architecture

The model consists of a [bert-base-japanese](https://huggingface.co/cl-tohoku/bert-base-japanese-v2) and two linear classification heads, one for `labels` and one for `detect`. `labels` predicts a specific edit transformation (`$KEEP`, `$DELETE`, `$APPEND_x`, etc), and `detect` predicts whether the token is `CORRECT` or `INCORRECT`. The results from the two are used to make a prediction. The predicted transformations are then applied to the errorful input sentence to obtain a corrected sentence.

Furthermore, in some cases, one pass of predicted transformations is not sufficient to transform the errorful sentence to the target sentence. Therefore, we repeat the process again on the result of the previous pass of transformations, until the model predicts that the sentence no longer contains incorrect tokens.

For more details about the model architecture and __iterative sequence tagging approach__, refer to the GECToR [article](https://www.grammarly.com/blog/engineering/gec-tag-not-rewrite/) or the [official implementation](https://github.com/grammarly/gector/blob/master/gector/seq2labels_model.py).

## Training
Install the requirements:
```
pip install -r requirements.txt
```

The model was trained in Colab with GPUs on each corpus with the following hyperparameters (default is used if unspecified):
```
python ./utils/combine.py
python ./utils/preprocess.py
bash train.sh
```

## Demo

Trained weights can be downloaded [here](https://drive.google.com/file/d/1nhWzDZnZKxLvqwYMLlwRNOkMK2aXv4-5/view?usp=sharing). The trained weights have been pre-trained on JaWiki and Lang8.

Extract `model.zip` to the `data/` directory. You should have the following folder structure:

```
gector-ja/
  utils/
    data/
      model/
        checkpoint
        model_checkpoint.data-00000-of-00001
        model_checkpoint.index
      ...
    main.py
    ...
```

After downloading and extracting the weights, the demo app can be run with the command `python main.py`.

You may need to `pip install flask` if Flask is not already installed.

## Evaluation

The model can be evaluated with `evaluate.py` on a parallel sentences corpus. The evaluation corpus used was [TMU Evaluation Corpus for Japanese Learners (TEC_JL)](https://github.com/koyama-aomi/TEC-JL), and the metric is GLEU score.

Using the model trained with the parameters described above, it achieved a GLEU score of around **0.83**, which appears to outperform the CNN-based method by Chollampatt and Ng, 2018 (state of the art on the CoNLL-2014 dataset prior to transformer-based models), that Koyama et al. 2020 chose to use in their paper.

#### TMU Evaluation Corpus for Japanese Learners (GEC dataset for Japanese)
| Method                    | GLEU  |
| ------------------------- | ----- |
| Chollampatt and Ng, 2018  | 0.739 |
| __gector_jsc (mine)__  | __0.83__  |

In this project GLEU score was used as in Koyama et al. 2020.

## Credit
[jonnyli1125
](https://github.com/jonnyli1125)
