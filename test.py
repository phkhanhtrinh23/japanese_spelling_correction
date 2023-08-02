import torch
from torch import nn
from underthesea import word_tokenize
from transformers import AutoModel, AutoTokenizer

class BaseModel(nn.Module):
    def __init__(self,
                 model_name_or_path,
                 max_length,
                 padding="max_length",
                 truncation=True):
        super(BaseModel, self).__init__()
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModel.from_pretrained(self.model_name_or_path)

    def forward(self,
                input_texts):
        input_encodings = self.tokenizer(input_texts,
                                         max_length=self.max_length,
                                         padding=self.padding,
                                         truncation=self.truncation,
                                         return_tensors='pt')
        
        print("Input encodings:", input_encodings)
        print("Input encodings shape:", input_encodings["input_ids"].size()) # torch.Size([batch_size, max_length]) = torch.Size([2, 256])

        output_embedding = self.model(**input_encodings)

        return output_embedding
    
if __name__ == "__main__":
    base_model = BaseModel(
                        model_name_or_path="vinai/phobert-base-v2",
                        max_length=256,                    
                    )
    input_texts = [
        "Tôi thích ăn cơm sườn",
        "Môi trường xung quanh chúng ta đang bị ô nhiễm",
    ]
    
    segment_input_texts = []
    for text in input_texts:
        output = word_tokenize(text, format="text")
        segment_input_texts.append(output)

    embeddings = base_model(segment_input_texts)

    print("Output embeddings shape:\n{}".format(embeddings[0].size())) # torch.Size([batch_size, max_length, embedding_size]) = torch.Size([2, 256, 768])