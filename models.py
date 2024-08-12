import torch.nn as nn
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class CaptionGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(CaptionGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = nn.Transformer(d_model=embed_size, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        features = features.unqueeze(1)
        output = self.transformer(features, embeddings)
        output = self.fc(output)
        return output


class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, captions):
        embeddings = self.embedding(captions)
        output, _ = self.lstm(embeddings)
        output = self.fc(output)
        return output


class TransformerGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers):
        super(TransformerGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = nn.Transformer(d_model=embed_size, nhead=num_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, captions):
        embeddings = self.embedding(captions)
        output = self.transformer(embeddings, embeddings)
        output = self.fc(output)
        return output


class FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(FeatureExtractor, self).__init__()
        resnet50 = models.resnet50(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet50.children()[:-1]))

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
        return features.squeeze()

    def extract_features(self, image_path, device='cpu'):
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0)  # adding batch dimension

        image = image.to(device)

        features = self.forward(image)

        return features.cpu().numpy()
