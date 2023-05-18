import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# Model1 : 1xBi-LSTM

class LSTMnet(nn.Module):
    def __init__(self, vocab_size, embedding_size, lstm_dim, hidden_dim, out_dim, drop):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_size,
                            lstm_dim,
                            num_layers=2,
                            dropout=drop,
                            batch_first=True,
                            bidirectional=True)
        
        # Fully connected layer for classif
        self.fc = nn.Sequential(nn.Linear(in_features=lstm_dim*2, out_features=hidden_dim),
                                nn.BatchNorm1d(hidden_dim),
                                nn.Sigmoid(),
                                nn.Linear(in_features=hidden_dim, out_features=out_dim),
                                nn.BatchNorm1d(out_dim),
                                nn.Sigmoid())
        
    # Forward function
    def forward(self, seq, seq_len):
        
        # first pass into embed layer
        embed = self.embedding(seq)
        
        # pack_pad the sequences is a good way to gate relevant information in padded sequences to train a RNN network 
        pack_padded = nn.utils.rnn.pack_padded_sequence(input=embed,
                                                        lengths=seq_len,
                                                        batch_first=True,
                                                        enforce_sorted=False) # because sequences are non sorted before
        # pass into lstm layer
        lstm_out, (h_n, c_n) = self.lstm(pack_padded)
        
        # concatenate into dim 1 outputs of all lstm cells to get more informations
        h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        
        # pass into fc
        out = self.fc(h_n)
        
        return out


    
############## Model 1 : 2xBi-LSTM

class LSTMnet2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm1_dim, dropout1, lstm2_dim, dropout2, num_layers, hidden_dim, out_dim):
        super().__init__()
        
        self.num_layers = num_layers
        self.lstm1_dim = lstm1_dim
        self.lstm2_dim = lstm2_dim
        
        # embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        
        # 2 bi-lstm in a sequential
        # 1st
        self.lstm1 = nn.LSTM(input_size = embedding_dim,
                              hidden_size = lstm1_dim,
                              num_layers = num_layers,
                              batch_first = True,
                              dropout = dropout1,
                              bidirectional = True)
        # 2nd    
        self.lstm2 = nn.LSTM(input_size = lstm1_dim*2,
                              hidden_size = lstm2_dim,
                              num_layers = num_layers,
                              batch_first = True,
                              dropout = dropout2,
                              bidirectional = True)
        
        # fully connected for classif
        self.fc = nn.Sequential(
            nn.Linear(in_features=lstm2_dim*2, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=out_dim),
            nn.Softmax(dim=1)
        )
        
    # Forward function
    def forward(self, seq, seq_len):
        
        # first pass into embed layer
        embed = self.embedding(seq)
        
        # pack_pad the sequences is a good way to get relevant information in padded sequences to train a RNN network 
        pack_padded = nn.utils.rnn.pack_padded_sequence(input=embed,
                                                        lengths=seq_len,
                                                        batch_first=True,
                                                        enforce_sorted=False) # because sequences are non sorted before
        
        # init hidden states to 0 for first lstm
        h1_0 = torch.zeros(self.num_layers*2, seq.size(0), self.lstm1_dim).to(seq.device)
        c1_0 = torch.zeros(self.num_layers*2, seq.size(0), self.lstm1_dim).to(seq.device)
        
        # pass into first lstm
        #lstm1_out, (h1_n, c1_n) = self.lstm1(embed, (h1_0, c1_0))
        lstm1_out, (h1_n, c1_n) = self.lstm1(pack_padded, (h1_0, c1_0))#, (h1_0, c1_0))  , (h1_n, c1_n)
        
        # init hidden states to 0 for 2nd lstm
        h2_0 = torch.zeros(self.num_layers*2, seq.size(0), self.lstm2_dim).to(seq.device)
        c2_0 = torch.zeros(self.num_layers*2, seq.size(0), self.lstm2_dim).to(seq.device)
        
        #pass into 2nd lstm
        lstm_out2, (h2_n, c2_n) = self.lstm2(lstm1_out, (h2_0, c2_0))
        
        
        # concatenate into dim 1 outputs of all lstm cells to get more informations
        h_n = torch.cat((h2_n[-2, :, :], h2_n[-1, :, :]), dim=1)
        
        # pass into fc
        out = self.fc(h_n)
        
        return out

