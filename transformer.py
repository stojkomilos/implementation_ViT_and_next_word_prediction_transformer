import torch.nn.functional as F
import torch.nn as nn
import math
import torch

class Transformer(torch.nn.Module):
    # this is encoder only or decoder only transformer. It cant have both encoders and decoder at the same time
    def __init__(self, nr_encoder_decoder_blocks, nr_heads, model_dim, max_input_tokens, nr_input_features, use_masked_attention):
        super().__init__()
        self.nr_heads = nr_heads
        self.model_dim = model_dim
        self.nr_encoders = nr_encoder_decoder_blocks

        if use_masked_attention:
            self.mask = torch.tril(torch.ones(max_input_tokens, max_input_tokens)) # we use lower triangular instead of upper triangular, because we want the token to intereact with itself
        else:
            self.mask = None

        self.encoders_decoders = [EncoderDecoderBlock(nr_heads=nr_heads, model_dim=model_dim, use_masked_attention=use_masked_attention, mask=self.mask) for i in range(nr_encoder_decoder_blocks)]

        self.embedding_transform = nn.Linear(in_features=nr_input_features, out_features=model_dim, bias=False)

        self.positional_embeddings = PositionalEmbeddings(model_dim, max_input_tokens)
    
    def forward(self, x):
        x_embedded = self.embedding_transform(x)

        x_mha_input = x_embedded + self.positional_embeddings(x_embedded)

        for i in range(self.nr_encoders):
            x_mha_input = self.encoders_decoders[i](x_mha_input)
        
        return x_mha_input

USE_LEARNED_EMBEDDIGNS = False
class PositionalEmbeddings(torch.nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))  # Shape: [d_model // 2]
        
        # Apply sin to even indices and cos to odd indices
        pe = torch.zeros(max_len, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices in the embedding
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices in the embedding

        # Register the positional encodings as a buffer, so they are not considered a model parameter
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: [1, max_len, d_model]
    
    def forward(self, x):
        return self.pe[:, :x.size(1), :]  # The size of the sequence (x.size(1)) is used

class MLP(torch.nn.Module):
    def __init__(self, model_dim, inter_dim, use_vit_mlp):
        super().__init__()

        self.model_dim = model_dim
        self.inter_dim = inter_dim

        if use_vit_mlp:
            # uses mlp from ViT paper
            self.mlp = nn.Sequential(
                nn.Linear(model_dim, inter_dim),
                nn.GELU(),
                nn.Linear(inter_dim, model_dim)
            )
        else:
            # uses mlp from original Attention is all you need paper
            self.mlp = nn.Sequential(
                nn.Linear(in_features=model_dim, out_features=inter_dim),
                nn.ReLU(),
                nn.Linear(in_features=inter_dim, out_features=model_dim)
            )

    def forward(self, x):
        x = self.mlp(x)
        # x = self.relu(self.layer1(x))
        # x = self.layer2(x)
        return x

class EncoderDecoderBlock(torch.nn.Module):
    # this block can be used if you use encoder only or decoder only architecture, but not if you have encoders and decoders
    def __init__(self, nr_heads, model_dim, use_masked_attention, mask=None):
        super().__init__()
        self.use_masked_attention = use_masked_attention
        self.mask = mask

        if use_masked_attention is False:
            assert mask is None
        else:
            assert mask is not None

        self.mha = MultiHeadAttention(nr_heads=nr_heads, model_dim=model_dim, use_masked_attention=use_masked_attention, mask=mask)
        self.layer_norm1 = LayerNorm(model_dim)
        self.layer_norm2 = LayerNorm(model_dim)
        # one encoder uses the same MLP for all embedding vectors, but different encoders use different MLPs
        self.mlp = MLP(model_dim=model_dim, inter_dim=model_dim*4, use_vit_mlp=True) # 4*model_dim because Attention is All you need did this

    def forward(self, x):

        debug_print = False

        # we remember the residual in /x/
        out = self.mha(x)
        out = self.layer_norm1(out)
        x = x + out

        # we remember the residual in /x/ again

        if debug_print:
            print(f'{x.shape=}')

        # position wise mlp, so that input to mlp is a single vector embedding
        out = self.mlp(x)

        if debug_print:
            print(f'{out.shape=}')

        out = self.layer_norm2(out)

        return x + out


class LayerNorm(torch.nn.Module):
    def __init__(self, input_shape, epsilon=1e-5):
        super().__init__()
        self.betas = torch.zeros(input_shape)
        self.gammas = torch.ones(input_shape)
        self.epsilon = epsilon
    
    def forward(self, x):
        mean = x.mean(dim=-1)
        std = x.std(dim=-1)

        mean = mean.unsqueeze(-1)
        std = std.unsqueeze(-1)

        normalized = (x-mean)/(std+self.epsilon)

        return normalized*self.gammas + self.betas

class MultiHeadAttention(torch.nn.Module):
    # this block can be used if you use encoder only or decoder only architecture, but not if you have encoders and decoders
    def __init__(self, nr_heads, model_dim, use_masked_attention, mask=None):
        super().__init__()

        self.model_dim = model_dim
        self.nr_heads = nr_heads
        assert model_dim % nr_heads == 0
        self.qk_dim = model_dim // nr_heads
        self.use_masked_attention = use_masked_attention
        self.mask = mask

        if use_masked_attention is False:
            assert mask is None
        else:
            assert mask is not None

        # we use nn.Linear instead of regular pytorch matrices because of backprop
        self.key_matrix = nn.Linear(in_features=model_dim, out_features=model_dim, bias=False)
        self.query_matrix = nn.Linear(in_features=model_dim, out_features=model_dim, bias=False)
        self.value_matrix = nn.Linear(in_features=model_dim, out_features=model_dim, bias=False)

        self.value_up_matrix = nn.Linear(in_features=model_dim, out_features=model_dim, bias=False)
    
        self.value_up_matrix = nn.Linear(in_features=model_dim, out_features=model_dim)

        self.qk_dimension_sqrt_tensor = torch.sqrt(torch.tensor(self.model_dim//self.nr_heads, dtype=torch.float32))

    def forward(self, x):

        batch_size, nr_input_embeddings, model_dim = x.shape

        debug_print = False

        assert model_dim == self.model_dim

        Q = self.query_matrix(x)
        K = self.key_matrix(x)
        V = self.value_matrix(x)

        if debug_print:
            print(f'pre - {Q.shape=}')
            print(f'pre - {K.shape=}')
            print(f'pre - {V.shape=}')

        K = K.view(batch_size, nr_input_embeddings, self.nr_heads, self.qk_dim).transpose(1, 2)
        Q = Q.view(batch_size, nr_input_embeddings, self.nr_heads, self.qk_dim).transpose(1, 2)
        V = V.view(batch_size, nr_input_embeddings, self.nr_heads, self.qk_dim).transpose(1, 2)

        if debug_print:
            print(f'post - {Q.shape=}')
            print(f'post - {K.shape=}')
            print(f'post - {V.shape=}')

        # dynamic_weights = F.softmax(qk / self.kq_dimension_sqrt_tensor, dim=1) # zelimo da radimo softmax po prvom redu, pa po drugom redu itd... 
        # dim=-1? !!!! pazi kod softmax dimenzije da te ne sjebe batch size i broj ulaznih tokena

        # qk.shape() = (batch_size, nr_heads, nr_input_embeddings, nr_input_embeddings) where the first of the two nr_input_embeddings shows the dynamic weight of how all other embeddings influence the first embedding (thats why we softmax over the -1 axis)
        qk = Q @ K.transpose(-2, -1) / self.qk_dimension_sqrt_tensor

        if self.use_masked_attention:
            qk = qk.masked_fill(~self.mask[:nr_input_embeddings, :nr_input_embeddings].bool(), float('-inf'))
            # print('qk:', qk)

        dynamic_weights = F.softmax(qk, dim=-1)
        # print('dynamic_weights=\n', dynamic_weights)

        if debug_print:
            print(f'{qk.shape=}')
            print(f'{dynamic_weights.shape=}')

        ret = dynamic_weights @ V

        if debug_print:
            print(f'{ret.shape=}')

        ret = ret.transpose(1, 2)

        if debug_print:
            print(f'{ret.shape=}')

        ret = ret.contiguous().view(x.shape)

        return self.value_up_matrix(ret)