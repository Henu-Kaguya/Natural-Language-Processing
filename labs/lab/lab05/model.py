import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, token_vocab, tag_vocab, batch_size,
                 dropout=0.5, embedding_dim=256,
                 hidden_dim=256, pretrained_embedding=None,
                 padding_idx=0, num_layers=1):
        super(BiLSTM_CRF, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.token_vocab = token_vocab
        self.tag_vocab = tag_vocab
        self.pad = self.token_vocab.pad_token

        self.tagset_size = len(tag_vocab)
        self.begin_tag_idx = tag_vocab.lookup_token('<start>')
        self.end_tag_idx = tag_vocab.lookup_token('<end>')

        if pretrained_embedding is None:
            self.word_embeds = nn.Embedding(len(self.token_vocab), embedding_dim)
        else:
            self.word_embeds = nn.Embedding(len(self.token_vocab), embedding_dim,
                                            _weight=pretrained_embedding)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=num_layers, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.transition = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))
        self.transition.data[self.begin_tag_idx, :] = -10000
        self.transition.data[:, self.end_tag_idx] = -10000

        self.hidden = self.init_hidden(num_layers, batch_size)

    def init_hidden(self, num_layers, batch_size):
        return (torch.randn(2 * num_layers, batch_size, self.hidden_dim // 2, device=self.device),
                torch.randn(2 * num_layers, batch_size, self.hidden_dim // 2, device=self.device))

    def _forward_alg(self, feats, mask):
        """Forward algorithm for CRF partition function

        Args:
            feats: [b_s, seq_len, tag_size]
            mask: [b_s, seq_len]
        Returns:
            [b_s] partition function scores
        """
        init_alphas = torch.full((feats.size(0), self.tagset_size), -10000., device=self.device)
        init_alphas[:, self.begin_tag_idx] = 0.

        forward_var_list = []
        forward_var_list.append(init_alphas)
        d = torch.unsqueeze(feats[:, 0], dim=1)
        for feat_index in range(1, feats.size(1)):
            n_unfinish = mask[:, feat_index].sum()
            d_uf = d[:n_unfinish]
            emit_and_transition = feats[:n_unfinish, feat_index].unsqueeze(dim=1) + self.transition
            log_sum = d_uf.transpose(1, 2) + emit_and_transition
            max_v = log_sum.max(dim=1)[0].unsqueeze(dim=1)
            log_sum = log_sum - max_v
            d_uf = max_v + torch.logsumexp(log_sum, dim=1).unsqueeze(dim=1)
            d = torch.cat((d_uf, d[n_unfinish:]), dim=0)
        d = d.squeeze(dim=1)
        max_d = d.max(dim=-1)[0]
        d = max_d + torch.logsumexp(d - max_d.unsqueeze(dim=1), dim=1)
        return d

    def _get_lstm_features(self, embedded_vec, seq_len):
        """Get emission scores from BiLSTM

        Args:
            embedded_vec: [max_seq_len, b_s, e_d]
            seq_len: [b_s]
        Returns:
            [b_s, seq_len, tag_size]
        """
        pack_seq = pack_padded_sequence(embedded_vec, seq_len)
        lstm_out, self.hidden = self.lstm(pack_seq)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_feats = self.hidden2tag(lstm_out)
        lstm_feats = self.dropout(lstm_feats)
        return lstm_feats

    def _score_sentence(self, feats, tags, mask):
        """Score the gold tag sequence

        Args:
            feats: [b_s, seq_len, tag_size]
            tags: [b_s, seq_len]
            mask: [b_s, seq_len]
        Returns:
            [b_s] gold path scores
        """
        score = torch.gather(feats, dim=2, index=tags.unsqueeze(dim=2)).squeeze(dim=2)
        score[:, 1:] += self.transition[tags[:, :-1], tags[:, 1:]]
        total_score = (score * mask.type(torch.float)).sum(dim=1)
        return total_score

    def _viterbi_decode(self, feats, mask, seq_len):
        """Viterbi decoding for finding best tag sequence

        Args:
            feats: [b_s, seq_len, tag_size]
            mask: [b_s, seq_len]
            seq_len: [b_s]
        Returns:
            scores, tag_sequences
        """
        batch_size = feats.size(0)
        tags = [[[i] for i in range(len(self.tag_vocab))]] * batch_size
        d = torch.unsqueeze(feats[:, 0], dim=1)
        for i in range(1, seq_len[0]):
            n_unfinished = mask[:, i].sum()
            d_uf = d[:n_unfinished]
            emit_and_transition = self.transition + feats[:n_unfinished, i].unsqueeze(dim=1)
            new_d_uf = d_uf.transpose(1, 2) + emit_and_transition
            d_uf, max_idx = torch.max(new_d_uf, dim=1)
            max_idx = max_idx.tolist()
            tags[:n_unfinished] = [[tags[b][k] + [j] for j, k in enumerate(max_idx[b])] for b in range(n_unfinished)]
            d = torch.cat((torch.unsqueeze(d_uf, dim=1), d[n_unfinished:]), dim=0)
        d = d.squeeze(dim=1)
        score, max_idx = torch.max(d, dim=1)
        max_idx = max_idx.tolist()
        tags = [tags[b][k] for b, k in enumerate(max_idx)]
        return score, tags

    def neg_log_likelihood(self, token_vec, tag_vec, seq_len):
        """Compute negative log likelihood loss"""
        mask = (token_vec != self.token_vocab.lookup_token(self.pad)).to(self.device)
        token_vec = token_vec.transpose(0, 1)
        embedded_vec = self.word_embeds(token_vec)
        feats = self._get_lstm_features(embedded_vec, seq_len)

        forward_score = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(feats, tag_vec, mask)
        return forward_score - gold_score

    def forward(self, token_vec, tag_vec, seq_len):
        """Forward pass: Viterbi decoding to find best path

        Args:
            token_vec: [b_s, max_seq_len]
            tag_vec: [b_s, max_seq_len]
            seq_len: [b_s]
        Returns:
            scores, tag_sequences
        """
        mask = (token_vec != self.token_vocab.lookup_token(self.pad)).to(self.device)
        token_vec = token_vec.transpose(0, 1)
        embedded_vec = self.word_embeds(token_vec)
        lstm_feats = self._get_lstm_features(embedded_vec, seq_len)

        mask = mask[:, :lstm_feats.size(1)]
        score, tag_seq = self._viterbi_decode(lstm_feats, mask, seq_len)
        return score, tag_seq

    @property
    def device(self):
        return self.word_embeds.weight.device
