import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# One can use any DL architecture for embedding including fully connected models, GRU, LSTM etc.
# The paper "Convolutional Embedding for Edit Distance" showed CNN-1D models can
# learn some interesting embeddings related to edit distance.
# This model is adapted from that paper.
class EmbeddingCNNNetwork(nn.Module):
    def __init__(self, string_helper, configs):
        super(EmbeddingCNNNetwork, self).__init__()
        self.embedding_dimension = configs.embedding_dimension
        self.string_helper = string_helper
        self.channel_size = configs.channel_size
        self.max_string_length = self.string_helper.max_string_length
        self.alphabet_size = self.string_helper.alphabet_size

        self.conv = nn.Sequential(
            nn.Conv1d(1, self.channel_size, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(self.channel_size, self.channel_size, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(self.channel_size, self.channel_size, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
        )

        # Size after pooling
        # 8 = 2*2*2 = one for each Conv1d layers
        self.flat_size = self.max_string_length // 8 * self.alphabet_size * self.channel_size
        self.fc1 = nn.Linear(self.flat_size, self.embedding_dimension)

    # Perform convoution to compute the embedding
    def forward(self, x):
        N = len(x)
        x = x.view(-1, 1, self.max_string_length)
        x = self.conv(x)
        x = x.view(N, self.flat_size)
        x = self.fc1(x)
        return x


def train_embedding_model(configs, train_loader, string_helper, load_model=False):
    model = EmbeddingCNNNetwork(string_helper, configs)
    # Comment this line during experimentation.
    #model = torch.jit.script(model)
    model = model.to(configs.device)
    if load_model:
        return model

    optimizer = optim.Adam(model.parameters(), lr=configs.lr)
    criterion = nn.TripletMarginLoss(margin=configs.margin)

    device = configs.device

    model.train()
    # for epoch in tqdm(range(configs.num_epochs), desc="Epochs"):
    for epoch in range(configs.num_epochs):
        # for epoch in range(epochs):
        running_loss = []
        # for step, (anchor, positive, negative) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        for step, (anchor, positive, negative) in enumerate(train_loader):
            # for step, (anchor, positive, negative) in enumerate(train_loader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.cpu().detach().numpy())
        # print("Epoch: {}/{} - Mean Running Loss: {:.8f}".format(epoch+1, configs.num_epochs, np.mean(running_loss)))

    return model
