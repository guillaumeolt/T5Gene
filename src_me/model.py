import torch
import torch.nn as nn
import torch.nn.functional as F
import selfies as sf
from torch.utils.data import Dataset
import numpy as np

# gradient clipping
def get_optim_params(model):
    return (p for p in model.vae.parameters() if p.requires_grad)

def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5): # https://github.com/haofuml/cyclical_annealing/blob/master/plot/plot_schedules.ipynb # for kl annealing
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_epoch):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L  

class SelfiesDataset(Dataset):
    def __init__(self, selfies_data, vocab, pad_to_len):
        self.selfies_data = selfies_data  # Assuming data is a list or numpy array of samples
        self.vocab = vocab
        self.pad_to_len = pad_to_len

    def __len__(self):
        return len(self.selfies_data)

    def __getitem__(self, idx):
        # Get SELFIES string
        selfies_data = self.selfies_data[idx]

        # Convert SELFIES strings to indices
        selfies_indices = [self.vocab[char] for char in sf.split_selfies(selfies_data)]
       
        # Pad sequences to ensure equal length
        padded_indices = selfies_indices + [0] * (self.pad_to_len - len(selfies_indices))

        # Convert SELFIES indices to PyTorch tensor
        selfies_tensor = torch.tensor(padded_indices)
        return selfies_tensor

class VAE(nn.Module):
    def __init__(self, vocab, vocab_size, embedding_dim, hidden_dim, latent_dim, max_length):
        super().__init__()

        self.vocabulary = vocab
        self.vocabulary_ids = {v: k for k, v in vocab.items()} # inverse vocabulary
        self.bos = vocab['[BOS]']
        self.eos = vocab['[EOS]']
        self.pad = vocab['[PAD]']
        self.max_length = max_length

        # Word embeddings layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.pad)
        self.embedding.weight.data.copy_(torch.eye(vocab_size, embedding_dim))

        # Encoder
        self.encoder = nn.GRU(embedding_dim, 256, num_layers=1, batch_first=True, dropout=0, bidirectional=True) # hidden dim encoder != decoder
        # The encoder outputs the mean and log-variance of the latent distribution
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.lat = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(latent_dim+embedding_dim, hidden_dim, num_layers=3, batch_first=True, dropout=0.2) # sequence-to-sequence model training loop with attention mechanism
        # The decoder outputs the logits of the vocabulary
        self.fc = nn.Linear(hidden_dim, max_length)


        # Grouping the model's parameters (fro gradient clipping)
        self.encoder_module = nn.ModuleList([
            self.encoder,
            self.fc_mu,
            self.fc_logvar
        ])
        self.decoder_module = nn.ModuleList([
            self.decoder,
            self.lat,
            self.fc
        ])
        self.vae = nn.ModuleList([
            self.embedding,
            self.encoder_module,
            self.decoder_module
        ])

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        """Do the VAE forward step

        :param x: list of tensors of longs, input sentence x
        :return: float, kl term component of loss
        :return: float, recon component of loss
        """

        # Encoder: x -> z, kl_loss
        z, kl_loss = self.forward_encoder(x)
        # Decoder: x, z -> recon_loss
        recon_loss = self.forward_decoder(x, z)
        
        return kl_loss, recon_loss

    def forward_encoder(self, x):
        """Encoder step, emulating z ~ E(x) = q_E(z|x)

        :param x: list of tensors of longs, input sentence x
        :return: (n_batch, d_z) of floats, sample of latent vector z
        :return: float, kl term component of loss
        """

        x = [self.embedding(i_x) for i_x in x]
        x = nn.utils.rnn.pack_sequence(x)

        _, h = self.encoder(x, None)

        h = h[-(1 + int(self.encoder.bidirectional)):] # bidirectional
        h = torch.cat(h.split(1), dim=-1).squeeze(0)

        mu, logvar = self.fc_mu(h), self.fc_logvar(h)

        eps = torch.randn_like(mu)
        z = mu + (logvar / 2).exp() * eps # reparameterization trick
        
        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()
        
        return z, kl_loss

    def forward_decoder(self, x, z):
        """Decoder step, emulating x ~ G(z)

        :param x: list of tensors of longs, input sentence x
        :param z: (n_batch, d_z) of floats, latent vector z
        :return: float, recon component of loss
        """
        lengths = [len(i_x) for i_x in x]

        x = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                      padding_value=self.pad)
        #print(x.shape)
        #print("x", x[:, 1:].contiguous().view(-1)[:10], x[:, 1:].contiguous().view(-1).shape)
        x_emb = self.embedding(x)

        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths,
                                                    batch_first=True)
        

        h_0 = self.lat(z)
        h_0 = h_0.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        output, _ = self.decoder(x_input, h_0)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.fc(output)
        #print(y.shape)
        #print("y", y[:, :-1].contiguous().view(-1, y.size(-1))[:10], y[:, :-1].contiguous().view(-1, y.size(-1)).shape )
        recon_loss = F.cross_entropy(
            y[:, :-1].contiguous().view(-1, y.size(-1)),
            x[:, 1:].contiguous().view(-1),
            ignore_index=self.pad
        )

        return recon_loss
    
    def tensor2selfies(self, tensor):
        # Remove BOS and EOS tokens from the tensor
        ids = [id_ for id_ in tensor.tolist() if id_ not in [self.bos, self.eos]]

        string_selfies = sf.encoding_to_selfies(
            encoding = ids,
            vocab_itos = self.vocabulary_ids,
            enc_type = "label"
        )

        return string_selfies
    
    def encode_decode(self, x, temp = 1.0 ):
        """
        encode and decode a batch of sequences with the vae

        :param temp: temperature of softmax for decoding (default: 1.0) # TODO : explore removing multinomial part ?
        """
        with torch.no_grad():
            n_batch = x.size(0) # get the batch size

            ### Encoder: x -> z
            z, _ = self.forward_encoder(x)
            z = z.to(self.device) # ensure tensor is on the same device

            ### Decoder: x, z -> x (seq to seq)
            z_0 = z.unsqueeze(1) # add a dimension to z

            # Initial values
            h = self.lat(z) # decoder linear layer to get initial hidden state
            h = h.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1) # repeat hidden state for each layer

            w = torch.tensor(self.bos, device=self.device).repeat(n_batch) # initialize the first word as <BOS>
            x = torch.tensor([self.pad], device=self.device).repeat(n_batch, self.max_length) # initialize the sequence with <PAD>

            x[:, 0] = self.bos # set the first word as <BOS>
            end_pads = torch.tensor([self.max_length], device=self.device).repeat(n_batch) # initialize the end pads
            eos_mask = torch.zeros(n_batch, dtype=torch.uint8, device=self.device) # initialize the end of sentence mask

            # Generating cycle --> sequence generation process, using sequence-to-sequence model
            for i in range(1, self.max_length): # model generates the sequence token by token, starting from the second position (index 1).
                x_emb = self.embedding(w).unsqueeze(1) # get the word embedding of the previous word. The resulting embedding is then unsqueezed along dimension 1 to match the expected input shape.
                x_input = torch.cat([x_emb, z_0], dim=-1) # Concatenating embeddings and latent variable. concatenates the embedding of the current token (x_emb) with the latent variable (z_0) along the last dimension. This concatenation likely provides the model with information from both the current token and the latent variable.

                o, h = self.decoder(x_input, h) # pass the concatenated input and the hidden state to the decoder. The decoder returns the output and the new hidden state.
                y = self.fc(o.squeeze(1)) # pass the output through the linear layer to get the logits of the vocabulary. The logits are then passed through a softmax layer to get the probability distribution over the vocabulary.
                y = F.softmax(y / temp, dim=-1) # apply softmax to the logits to get the probability distribution over the vocabulary. This step controls the "sharpness" of the probability distribution over the vocabulary.
                
                #_, w = torch.max(y, dim=-1) # get the word with the highest probability. The word with the highest probability is assigned to the next position in the sequence. results in only selecting [C] 
                w = torch.multinomial(y, 1)[:, 0] # sample the next token from the probability distribution. The sampled token is then assigned to the next position in the sequence. This sampling step introduces randomness into the generation process.
                x[~eos_mask, i] = w[~eos_mask] # assign the sampled token to the next position in the sequence. The token is only assigned if the end-of-sentence token has not been sampled.
                i_eos_mask = ~eos_mask & (w == self.eos) # check if the end-of-sentence token has been sampled. If the end-of-sentence token has been sampled, the position of the end-of-sentence token is recorded.
                end_pads[i_eos_mask] = i + 1 # record the position of the end-of-sentence token. The position is recorded as the current position plus one to account for the zero-based indexing.
                eos_mask = eos_mask | i_eos_mask # update the end-of-sentence mask. The mask is updated to include the positions where the end-of-sentence token has been sampled.
            
            # Converting `x` to list of tensors
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :end_pads[i]])

            return [self.tensor2selfies(i_x) for i_x in new_x]

    def sample_z_prior(self, n_batch): # TODO : clean this function
        """Sampling z ~ p(z) = N(0, I)

        :param n_batch: number of batches
        :return: (n_batch, d_z) of floats, sample of latent z
        """

        return torch.randn(n_batch, self.fc_mu.out_features,
                           device=self.embedding.weight.device)
    
    def sample(self, n_batch, max_len=100, z=None, temp=1.0): # TODO : clean this function
        """Generating n_batch samples in eval mode (`z` could be
        not on same device)

        :param n_batch: number of sentences to generate
        :param max_len: max len of samples
        :param z: (n_batch, d_z) of floats, latent vector z or None
        :param temp: temperature of softmax
        :return: list of tensors of strings, samples sequence x
        """
        with torch.no_grad():
            if z is None:
                z = self.sample_z_prior(n_batch)
            z = z.to(self.device)
            z_0 = z.unsqueeze(1)

            # Initial values
            h = self.decoder_lat(z)
            h = h.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
            w = torch.tensor(self.bos, device=self.device).repeat(n_batch)
            x = torch.tensor([self.pad], device=self.device).repeat(n_batch,
                                                                    max_len)
            x[:, 0] = self.bos
            end_pads = torch.tensor([max_len], device=self.device).repeat(
                n_batch)
            eos_mask = torch.zeros(n_batch, dtype=torch.uint8,
                                   device=self.device)

            # Generating cycle
            for i in range(1, max_len):
                x_emb = self.x_emb(w).unsqueeze(1)
                x_input = torch.cat([x_emb, z_0], dim=-1)

                o, h = self.decoder(x_input, h)
                y = self.decoder_fc(o.squeeze(1))
                y = F.softmax(y / temp, dim=-1)

                w = torch.multinomial(y, 1)[:, 0]
                x[~eos_mask, i] = w[~eos_mask]
                i_eos_mask = ~eos_mask & (w == self.eos)
                end_pads[i_eos_mask] = i + 1
                eos_mask = eos_mask | i_eos_mask

            # Converting `x` to list of tensors
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :end_pads[i]])

            print(new_x)
            return [self.tensor2selfies(i_x) for i_x in new_x]
