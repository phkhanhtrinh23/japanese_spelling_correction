from model import GEC
from unidecode import unidecode
import unicodedata

class JGEC():
    def __init__(self, 
                 weights_path="./utils/data/model/model_checkpoint", 
                 vocab_dir="./utils/data/output_vocab",
                 transforms_file="./utils/data/transform.txt"):
        self.vocab_dir = vocab_dir
        self.transforms_file = transforms_file
        self.weights_path = weights_path
        
        self.gec = GEC(vocab_path=self.vocab_dir, 
                       verb_adj_forms_path=self.transforms_file,
                       pretrained_weights_path=self.weights_path)

    def __call__(self, source_sents, batch_size=64):
        new_source_sents = []
        for src_sent in source_sents:
            if isinstance(src_sent, str):
                converted_sentence = unicodedata.normalize('NFKC', src_sent).replace(' ', '')
                new_source_sents.append(converted_sentence)
        source_sents = new_source_sents
        
        source_batches = [source_sents[i:i + batch_size]
                          for i in range(0, len(source_sents), batch_size)]
        pred_tokens = []
        for i, source_batch in enumerate(source_batches):
            pred_batch = self.gec.correct(source_batch)
            pred_batch_tokens = [sent for sent in pred_batch]
            pred_tokens.extend(pred_batch_tokens)

        return pred_tokens

if __name__ == '__main__':
    obj = JGEC()
    source_sents = ["そして10時くらいに、喫茶店でレーシャルとジョノサンとベルに会いました",
                    "一緒にコーヒーを飲みながら、話しました。",
                    "来週、レーシャルと私はメルボンに行くつもりです。",
                    "昔の学校の友達が新築祝いパーティを開くことになっています。",
                    "そしてその予定を話しました。",
                    "そのあと、スポットライトと言うクラフトの店を見に行きました。",
                    "24日から26日まではavconと言うアニメとテレビゲームの大会です。",
                    "私はコスプレをしたいです。",
                    "だからコスプレの生地を探しました。",
                    "イチゴとチェリーの模様の生地を買いました。",
                    "かわいいくて、安い、素敵な生地です。"]

    res = obj(source_sents)
    
    print("Results:", res)
