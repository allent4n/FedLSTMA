import torch.nn as nn
import torch
from parser import args_parser


class FFNN(nn.Module):
   def __init__(self, seq_len, n_features, hidden_dim=64):
       super(FFNN, self).__init__()

       self.seq_len = seq_len
       self.n_features = n_features
       self.hidden_dim = hidden_dim

       self.fc1 = nn.Linear(seq_len * n_features, hidden_dim)
       self.fc2 = nn.Linear(hidden_dim, seq_len * n_features)


   def forward(self, x):
       x = x.reshape(-1, self.seq_len * self.n_features)
       x = nn.functional.relu(self.fc1(x))
       x = nn.functional.relu(self.fc2(x))
       x = x.reshape(self.seq_len, self.n_features)

       return x

class RNN(nn.Module):

    def __init__(self, seq_len, n_features, hidden_dim=64):
        super(RNN, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.hidden_dim = hidden_dim

        self.rnn = nn.RNN(
            input_size=n_features,
            hidden_size=n_features,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))

        x, _ = self.rnn(x) # RNN
        #x, (_, _) = self.rnn(x) # LSTM

        x = x.reshape((-1, self.n_features))
        #x = self.output_layer(x)

        return x.reshape(self.seq_len, self.n_features)

class GRU(nn.Module):

    def __init__(self, seq_len, n_features, hidden_dim=64):
        super(GRU, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.hidden_dim = hidden_dim

        self.rnn = nn.GRU(
            input_size=n_features,
            hidden_size=n_features,
            num_layers=1,
            batch_first=True
        )


    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))

        x, _ = self.rnn(x) # GRU
        x = x.reshape((-1, self.n_features))
        #x = self.output_layer(x)

        return x.reshape(self.seq_len, self.n_features)

class LSTM(nn.Module):

    def __init__(self, seq_len, n_features, hidden_dim=64):
        super(LSTM, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.hidden_dim = hidden_dim

        self.rnn = nn.LSTM(
            input_size=n_features,
            hidden_size=n_features,
            num_layers=1,
            batch_first=True
        )
    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))

        x, (_, _) = self.rnn(x) # LSTM
        x = x.reshape((-1, self.n_features))
        #x = self.output_layer(x)

        return x.reshape(self.seq_len, self.n_features)








#################
#### Encoder ####
#################
class LSTM_Encoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(LSTM_Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )


    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))

        #x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn1(x)

        return hidden_n.reshape((self.n_features, self.embedding_dim))

#################
#### Decoder ####
#################
class LSTM_Decoder(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(LSTM_Decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))

    x, (hidden_n, cell_n) = self.rnn1(x)
    #x, (hidden_n, cell_n) = self.rnn2(x)

    x = x.reshape((self.seq_len, self.hidden_dim))

    return self.output_layer(x)

#################
## AutoEncoder ##
#################
args = args_parser()
device = torch.device("cpu")
#device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

class LSTMRecurrentAutoencoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(LSTMRecurrentAutoencoder, self).__init__()

    self.encoder = LSTM_Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = LSTM_Decoder(seq_len, embedding_dim, n_features).to(device)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)

    return x







#################
#### Encoder ####
#################
class RNN_Encoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RNN_Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.RNN(
            input_size=n_features,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))

        #x, _ = self.rnn1(x)
        x, hidden_n = self.rnn1(x)

        return hidden_n.reshape((self.n_features, self.embedding_dim))

#################
#### Decoder ####
#################
class RNN_Decoder(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(RNN_Decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.RNN(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))

    x, hidden_n = self.rnn1(x)
    #x, hidden_n = self.rnn2(x)

    x = x.reshape((self.seq_len, self.hidden_dim))

    return self.output_layer(x)

#################
## AutoEncoder ##
#################

class RNNRecurrentAutoencoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(RNNRecurrentAutoencoder, self).__init__()

    self.encoder = RNN_Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = RNN_Decoder(seq_len, embedding_dim, n_features).to(device)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)

    return x



#################
#### Encoder ####
#################
class GRU_Encoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(GRU_Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.GRU(
            input_size=n_features,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))

        #x, _ = self.rnn1(x)
        x, hidden_n = self.rnn1(x)

        return hidden_n.reshape((self.n_features, self.embedding_dim))

#################
#### Decoder ####
#################
class GRU_Decoder(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(GRU_Decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.GRU(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))

    x, hidden_n = self.rnn1(x)
    #x, hidden_n = self.rnn2(x)

    x = x.reshape((self.seq_len, self.hidden_dim))

    return self.output_layer(x)

#################
## AutoEncoder ##
#################

class GRURecurrentAutoencoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(GRURecurrentAutoencoder, self).__init__()

    self.encoder = GRU_Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = GRU_Decoder(seq_len, embedding_dim, n_features).to(device)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)

    return x