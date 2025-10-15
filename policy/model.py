import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEmbedding1D(nn.Module):
  def __init__(self, d_model, seq_length, dropout):
    super().__init__()
    self.d_model = d_model
    self.seq_length = seq_length
    pe = torch.zeros(self.seq_length, d_model)
    position = torch.arange(0, self.seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
    return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, features, eps=10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
      mean = x.mean(dim=-1, keepdim=True)
      std = x.std(dim=-1, keepdim=True)
      return self.alpha * (x - mean)/torch.sqrt(std * std + self.eps) + self.bias


class SwishGLU(nn.Module):
    def __init__(self, d_model, d_internal, dropout):
        super().__init__()
        self.proj = nn.Linear(d_model, 2 * d_internal)
        self.act = nn.SiLU()
        self.out = nn.Linear(d_internal, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        x = self.out(x1 * self.act(x2))
        return self.dropout(x)


class MultiheadAttentionBlock(nn.Module):
  def __init__(self, d_model, h, dropout):
    super().__init__()
    self.d_model = d_model
    self.h = h
    assert d_model % h ==0
    self.d_k = d_model // h
    self.Q = nn.Linear(d_model, d_model, bias=False)
    self.K = nn.Linear(d_model, d_model, bias=False)
    self.V = nn.Linear(d_model, d_model, bias=False)
    self.W = nn.Linear(d_model, d_model, bias=False)

    self.dropout = nn.Dropout(dropout)

  @staticmethod
  def attention(query, key, value, mask, dropout):
    d_k = query.shape[-1]
    attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
    if mask is not None:
      attention_scores.masked_fill_(mask == 0, -1e9)

    attention_scores= attention_scores.softmax(dim=-1)
    if dropout:
      attention_scores = dropout(attention_scores)

    return attention_scores @ value, attention_scores

  def forward(self, q, k, v, mask):
    query = self.Q(q) # [B seq_len d_model]
    key = self.K(k)
    value = self.V(v)
    # [B seq_len h d_k] -> [B h seq_len d_k]
    query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
    key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
    value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

    x, self.attention_scores = MultiheadAttentionBlock.attention(query, key, value, mask, self.dropout)

    # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
    x = x.transpose(1,2).contiguous().view(query.shape[0], query.shape[2], self.d_model)
    return self.W(x)


class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout):
      super().__init__()
      self.dropout = nn.Dropout(dropout)
      self.norm = LayerNormalization(d_model)

    def forward(self, x, sublayer):
      return self.norm(x + self.dropout(sublayer(x)))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, h, d_internal, dropout):
      super().__init__()
      self.self_attention = MultiheadAttentionBlock(d_model, h, dropout)
      self.ffnn = SwishGLU(d_model, d_internal, dropout)
      self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
      x = self.residual_connections[0](x, lambda val: self.self_attention(val, val, val, src_mask))
      return self.residual_connections[1](x, self.ffnn)


class Encoder(nn.Module):
    def __init__(self, d_model, h, d_internal, dropout, encoder_num):
      super().__init__()
      self.layers = nn.ModuleList([EncoderBlock(d_model, h, d_internal, dropout) for _ in range(2)])
      self.norm = LayerNormalization(d_model)

    def forward(self, x, mask=None):
      for layer in self.layers:
        x = layer(x, mask)
      return self.norm(x)


class DecoderBlock(nn.Module):
  def __init__(self, d_model, h, d_internal, dropout):
    super().__init__()
    self.self_attention = MultiheadAttentionBlock(d_model, h, dropout)
    self.cross_attention = MultiheadAttentionBlock(d_model, h, dropout)
    self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])
    self.ffnn = SwishGLU(d_model, d_internal, dropout)

  def forward(self, x, encoder_output, src_mask, tgt_mask):
    x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
    x = self.residual_connections[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
    x = self.residual_connections[2](x, self.ffnn)
    return x

class Decoder(nn.Module):
    def __init__(self, d_model, h, d_internal, dropout, decoder_num):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(d_model, h, d_internal, dropout) for _ in range(decoder_num)])
        self.norm = LayerNormalization(d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
  def __init__(self, d_model, actions_dim):
    super().__init__()
    self.proj = nn.Linear(d_model, actions_dim)

  def forward(self, x):
    return self.proj(x)


class Trener(nn.Module):
    def __init__(self, vision_encoder, config):
      super().__init__()
      self.actions_dim = config.actions_dim
      self.d_model = config.d_model
      self.img_features = config.img_features
      self.dropout = config.dropout
      self.config = config
      self.vision_encoder = vision_encoder
      for param in self.vision_encoder.parameters():
        param.requires_grad = False
      self.encoder = Encoder(config.d_model, config.h, config.d_internal, config.dropout, config.encoder_num)
      self.decoder = Decoder(config.d_model, config.h, config.d_internal, config.dropout, config.decoder_num)
      self.proj = ProjectionLayer(config.d_model, config.actions_dim)

      self.state_proj = nn.Linear(self.actions_dim, self.d_model)
      self.img_proj = nn.Linear(self.img_features, self.d_model)

      self.state_token = nn.Parameter(torch.randn(self.d_model))
      self.img_token = nn.Parameter(torch.randn(self.d_model))

      self.actions_embed = PositionalEmbedding1D(self.d_model, self.config.prediction_horizon, self.dropout)



    def forward(self, x):
      """
      x = {
        'index':
        'robot_state':
        'actions':
        'cam_key1':
        'cam_key2'
        ....
      }
      """
      feats = []
      with torch.no_grad():
          for cam_key in self.config.cam_keys:
            out = self.vision_encoder(x[cam_key]).last_hidden_state   # [B, patch_num, img_feat]
            feats.append(out)

      # [B, cam_num*patch_num, img_feat]
      images_tokens = self.img_proj(torch.cat(feats, dim=1))
      # [B, cam_num*patch_num, d_model]
      images_tokens += self.img_token.unsqueeze(0).unsqueeze(0)
      # robot_state [B, actions_dim]
      robot_state_tokens = self.state_proj(x['robot_state'])
      # robot_state [B, d_model]
      robot_state_tokens += self.state_token.unsqueeze(0)
      # robot_state [B, 1, d_model]
      robot_state_tokens = robot_state_tokens.unsqueeze(1)
      # encoder_input [B, cam_num*patch_num + 1, d_model]
      encoder_input = torch.cat([robot_state_tokens, images_tokens], dim=1)
      encoder_output = self.encoder(encoder_input)

      B = x['robot_state'].shape[0]
      decoder_input = torch.zeros(B, self.config.prediction_horizon, self.d_model, device=x['robot_state'].device)
      decoder_input = self.actions_embed(decoder_input)
      decoder_output = self.decoder(decoder_input, encoder_output)
      return self.proj(decoder_output)
