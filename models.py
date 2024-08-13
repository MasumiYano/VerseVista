import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer, GPT2LMHeadModel

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(FeatureExtractor, self).__init__()
        resnet50 = models.resnet50(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet50.children()[:-1]))

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
        return features.squeeze()

    def extract_features(self, image_path):
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform(image).unsqueeze(0)  # Adding a batch size

        image = image.to(device)

        features = self.forward(image)

        return features.numpy()


class CaptionGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_layers, output_size, incoming_features=2048):
        super(CaptionGenerator, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embedding_size
        self.num_layers = num_layers
        self.incoming_features = incoming_features
        self.output_size = output_size

        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        self.encoder = nn.Linear(in_features=self.incoming_features, out_features=self.embed_size)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.embed_size, nhead=8),
            num_layers=self.num_layers
        )
        self.fc = nn.Linear(in_features=self.embed_size, out_features=self.output_size)

    def forward(self, image_features, captions):
        embedded_captions = self.embed(captions)
        encoded_image = self.encoder(image_features).unsqueeze(0)  # Adding batch dimension
        transformer_output = self.transformer(embedded_captions, encoded_image)
        output = self.fc(transformer_output)
        return output


class RNNPoemGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, output_length=100):
        super(RNNPoemGenerator, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_length

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        self.rnn = nn.RNN(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

    def forward(self, captions):
        embeddings = self.embedding(captions)
        output, _ = self.rnn(embeddings)
        output = self.fc(output)
        return output


class LSTMPoemGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, output_length=100):
        super(LSTMPoemGenerator, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_length

        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

    def forward(self, captions):
        embeddings = self.embed(captions)
        output, _ = self.lstm(embeddings)
        output = self.fc(output)
        return output


class TransformerPoemGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, output_length=100):
        super(TransformerPoemGenerator, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.output_size = output_length

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        self.transformer = nn.Transformer(
            d_model=self.embed_size,
            nhead=self.num_heads,
            num_encoder_layers=self.num_layers,
            num_decoder_layers=self.num_layers
        )
        self.fc = nn.Linear(in_features=self.embed_size, out_features=self.output_size)

    def forward(self, captions):
        embeddings = self.embedding(captions)
        output = self.transformer(embeddings)
        output = self.fc(output)
        return output


class GPT2PoemGenerator(nn.Module):
    def __init__(self, pretrained_model_name='gpt2'):
        super(GPT2PoemGenerator, self).__init__()
        self.pretrained_model_name = pretrained_model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.pretrained_model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.pretrained_model_name)

    def forward(self, captions):
        inputs = self.tokenizer(captions, return_tensors='pt', padding=True)
        outputs = self.model(**inputs, labels=inputs['input_ids'])
        return outputs.logits

    def generate_poem(self, captions, max_length=100):
        inputs = self.tokenizer(captions, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=max_length, num_return_sequence=1)
        poem = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return poem