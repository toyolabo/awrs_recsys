# Adapted from https://github.com/taoqi98/CAUM/blob/main/Code/Models.py
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from newsreclib.models.components.layers.attention import DenseAttention
from newsreclib.models.components.encoders.news.comp_pop_pred import AvoidanceAwareRelevancePredictor


class AvoidanceAwareUserEncoder(nn.Module):
    """Implements an avoidance-aware user encoder that extends CAUM with user engagement patterns.

    This encoder enhances the CAUM architecture by incorporating user engagement patterns and temporal dynamics
    to model both content relevance and user avoidance behaviors in news recommendation.

    For further about CAUM model, please refer to the `paper <https://dl.acm.org/doi/abs/10.1145/3477495.3531778>`
    
    For more information on how the AvoidanceAwareUserEncoder works please refer to the original AWRS paper.

    Attributes:
        news_embed_dim:
            The number of features in the news vector.
        num_filters:
            The number of output features in the first linear layer.
        dense_att_hidden_dim1:
            The number of output features in the first hidden state of the ``DenseAttention``.
        dense_att_hidden_dim2:
            The number of output features in the second hidden state of the ``DenseAttention``.
        user_vector_dim:
            The number of features in the user vector.
        num_heads:
            The number of heads in the ``MultiheadAttention``.
        dropout_probability:
            Dropout probability.
        usr_eng_num_emb:
            The number of unique user engagement patterns to embed.
        usr_eng_emb_dim:
            The dimension of the user engagement embedding vectors.
        matrix_size:
            The size of the engagement pattern matrix (typically avoidance_size * epistemic_size).
        time_emb_dim:
            The dimension of the temporal embedding vectors.
        add_relevance_control:
            Whether to include the relevance control mechanism (default: False).
        add_avoidance_awareness:
            Whether to include the avoidance awareness mechanism (default: False).
    """

    def __init__(
        self,
        news_embed_dim: int,
        num_filters: int,
        dense_att_hidden_dim1: int,
        dense_att_hidden_dim2: int,
        user_vector_dim: int,
        num_heads: int,
        dropout_probability: float,
        usr_eng_num_emb: int,
        usr_eng_emb_dim: int,
        matrix_size: int,
        time_emb_dim: int,
        add_relevance_control: Optional[bool] = False,
        add_avoidance_awareness: Optional[bool] = False,
    ) -> None:
        super().__init__()

        if not isinstance(news_embed_dim, int):
            raise ValueError(
                f"Expected keyword argument `news_embed_dim` to be an `int` but got {news_embed_dim}"
            )

        if not isinstance(num_filters, int):
            raise ValueError(
                f"Expected keyword argument `num_filters` to be an `int` but got {num_filters}"
            )

        if not isinstance(dropout_probability, float):
            raise ValueError(
                f"Expected keyword argument `dropout_probability` to be a `float` but got {dropout_probability}"
            )

        # initialize
        self.dropout1 = nn.Dropout(p=dropout_probability)
        self.dropout2 = nn.Dropout(p=dropout_probability)
        self.dropout3 = nn.Dropout(p=dropout_probability)

        self.add_relevance_control = add_relevance_control
        self.add_avoidance_awareness = add_avoidance_awareness

        if self.add_avoidance_awareness:
            self.matrix_size = matrix_size
            self.user_eng_layer = nn.Embedding(usr_eng_num_emb, usr_eng_emb_dim)
            

        # -- Dense Linear Layer 1
        if self.add_avoidance_awareness:
            in_features_linear1 = news_embed_dim*4 + usr_eng_emb_dim*4
        else:
            in_features_linear1 = news_embed_dim * 4
        self.linear1 = nn.Linear(
            in_features=in_features_linear1, out_features=num_filters)
        
        # -- Dense Linear Layer 2
        if self.add_avoidance_awareness:
            in_features_linear2 = news_embed_dim*2 + usr_eng_emb_dim*2
        else:
            in_features_linear2 = news_embed_dim * 2
        self.linear2 = nn.Linear(
            in_features=in_features_linear2, out_features=user_vector_dim)

        # -- Dense Linear Layer 3
        if self.add_avoidance_awareness:
            out_features_linear3 = user_vector_dim + usr_eng_emb_dim
        else:
            out_features_linear3 = user_vector_dim
        self.linear3 = nn.Linear(
            in_features=num_filters + user_vector_dim, out_features=out_features_linear3
        )

        # -- Dense attention
        if self.add_avoidance_awareness:
            input_dim_dense_att = user_vector_dim * 2 + usr_eng_emb_dim * 2
        else:
            input_dim_dense_att = user_vector_dim * 2
        self.dense_att = DenseAttention(
            input_dim=input_dim_dense_att,
            hidden_dim1=dense_att_hidden_dim1,
            hidden_dim2=dense_att_hidden_dim2,
        )
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=user_vector_dim, num_heads=num_heads
        )

        if self.add_relevance_control:
            # initialize avoidance aware relevance predictor
            self.aw_rel_scores = AvoidanceAwareRelevancePredictor(
                news_embed_dim=news_embed_dim,
                time_emb_dim=time_emb_dim,
                usr_eng_num_emb=usr_eng_num_emb,
                usr_eng_emb_dim=usr_eng_emb_dim,
                matrix_size=matrix_size
            )

            # initialize personalized aggregator
            if self.add_avoidance_awareness:
                gate_eta_dim = news_embed_dim + usr_eng_emb_dim
            else:
                gate_eta_dim = news_embed_dim
            self.gate_eta = nn.Linear(gate_eta_dim, 1)

            # sigmoid function
            self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        hist_news_vector: torch.Tensor,
        cand_news_vector: torch.Tensor,
        hist_av_idx: torch.Tensor,
        hist_epi_idx: torch.Tensor,
        cand_av_idx: torch.Tensor,
        cand_epi_idx: torch.Tensor,
        cand_time_elapsed: torch.Tensor,
        cand_ctr: torch.Tensor,
    ) -> torch.Tensor:

        if self.add_avoidance_awareness:
            hist_user_eng_indices = hist_epi_idx * self.matrix_size + hist_av_idx
            hist_user_eng_emb = self.user_eng_layer(hist_user_eng_indices)
            hist_news_vector = torch.cat(
                (hist_news_vector, hist_user_eng_emb), dim=-1)

        if self.add_relevance_control:
            aw_relevance_score = self.aw_rel_scores(
                news_emb=cand_news_vector,
                time_elapsed=cand_time_elapsed,
                ctr=cand_ctr,
                av_idx=cand_av_idx,
                epi_idx=cand_epi_idx,
            )

        if self.add_avoidance_awareness:
            # Add user eng embedding
            cand_user_eng_indices = cand_epi_idx * self.matrix_size + cand_av_idx
            cand_user_eng_emb = self.user_eng_layer(cand_user_eng_indices)
            cand_news_vector = torch.cat(
                (cand_news_vector, cand_user_eng_emb), dim=-1)

        cand_news_vector = self.dropout1(cand_news_vector)
        hist_news_vector = self.dropout2(hist_news_vector)

        repeated_cand_news_vector = cand_news_vector.unsqueeze(dim=1).repeat(
            1, hist_news_vector.shape[1], 1
        )

        # -- candi-cnn
        hist_news_left = torch.cat(
            [hist_news_vector[:, -1:, :], hist_news_vector[:, :-1, :]], dim=-2
        )
        hist_news_right = torch.cat(
            [hist_news_vector[:, 1:, :], hist_news_vector[:, :1, :]], dim=-2
        )
        hist_news_cnn = torch.cat(
            [hist_news_left, hist_news_vector, hist_news_right,
                repeated_cand_news_vector], dim=-1
        )

        hist_news_cnn = self.linear1(hist_news_cnn)

        # -- candi-selfatt
        hist_news = torch.cat(
            [repeated_cand_news_vector, hist_news_vector], dim=-1)
        hist_news = self.linear2(hist_news)
        hist_news_self, _ = self.multihead_attention(
            hist_news, hist_news, hist_news)

        hist_news_all = torch.cat([hist_news_cnn, hist_news_self], dim=-1)
        hist_news_all = self.dropout3(hist_news_all)
        hist_news_all = self.linear3(hist_news_all)

        # -- candi-att
        attention_vector = torch.cat([hist_news_all, repeated_cand_news_vector], dim=-1)
        attention_score = self.dense_att(attention_vector)
        attention_score = attention_score.squeeze(dim=-1)
        attention_score = F.softmax(attention_score, dim=-1)

        user_vector = torch.bmm(attention_score.unsqueeze(dim=1), hist_news_all).squeeze(dim=1)

        scores = torch.bmm(
            cand_news_vector.unsqueeze(dim=1), user_vector.unsqueeze(dim=-1)
        ).flatten()

        if self.add_relevance_control:
            eta = self.sigmoid(self.gate_eta(user_vector)).squeeze(-1)

            scores = (1 - eta) * aw_relevance_score + eta * scores

        return scores
