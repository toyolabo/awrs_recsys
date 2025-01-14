from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from newsreclib.models.components.encoders.time.time2vec import Time2VecSineActivation


class FeatureAttention(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super(FeatureAttention, self).__init__()
        self.query = nn.Linear(emb_dim, hidden_dim)
        self.key = nn.Linear(emb_dim, hidden_dim)
        self.value = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_scores = F.softmax(attention_scores, dim=-1)
        context_vector = torch.matmul(attention_scores, values)
        return context_vector


class WeightedSumPooling(nn.Module):
    def __init__(self, emb_dim):
        super(WeightedSumPooling, self).__init__()
        self.weights = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        weighted_x = x * self.weights
        return weighted_x


class PopularityPredictor(nn.Module):
    def __init__(self, input_dim):
        super(PopularityPredictor, self).__init__()
        self.score_predictor = nn.Linear(input_dim, 1)

    def forward(self, x):
        scores = self.score_predictor(x)
        return scores.squeeze(-1)


class AvoidanceAwareRelevancePredictor(nn.Module):
    """Avoidance Aware Relevance Predictor"""

    def __init__(
        self,
        news_embed_dim: int,
        time_emb_dim: int,
        usr_eng_emb_dim: int,
        usr_eng_num_emb: int,
        matrix_size: int,
    ) -> None:
        super(AvoidanceAwareRelevancePredictor, self).__init__()

        # dense layers
        self.dense_news = nn.Linear(news_embed_dim, 1)
        self.gate = nn.Linear(time_emb_dim + usr_eng_emb_dim + news_embed_dim, 1)
        self.time_encoder = Time2VecSineActivation(output_dim=time_emb_dim)  # TODO: test with CosineActivation function as well
        self.dense_news_ctx = nn.Linear(time_emb_dim + usr_eng_emb_dim, 1)

        # User Engagement Layer
        self.user_eng_layer = nn.Embedding(usr_eng_num_emb, usr_eng_emb_dim)
        self.matrix_size = matrix_size

        # activation function
        self.sigmoid = nn.Sigmoid()

        # trainable weights
        self.w_ctr = nn.Parameter(torch.rand(1))  # ctr weight
        self.w_r_hat = nn.Parameter(torch.rand(1))  # relevance weight

    def forward(
        self,
        news_emb: torch.Tensor,
        time_elapsed: torch.Tensor,
        ctr: torch.Tensor,
        av_idx: torch.Tensor,
        epi_idx: torch.Tensor,
    ) -> torch.Tensor:

        # Compute user engagement embedding 
        user_eng_indices = epi_idx * self.matrix_size + av_idx
        user_eng_emb = self.user_eng_layer(user_eng_indices)

        # Compute time elapsed embeddings
        te_emb = self.time_encoder(time_elapsed)
        
        # Concat time elapsed with user eng
        news_ctx = torch.cat((user_eng_emb, te_emb), dim=-1)

        # Compute relevance influenced by information conveyed
        rel_hat_ic = self.dense_news(news_emb)

        # Compute relevance influenced by news context
        rel_hat_nctx = self.dense_news_ctx(news_ctx)

        # Combine time and news representations
        concat_emb = torch.cat((news_emb, news_ctx), dim=-1)

        # Compute theta
        theta = self.sigmoid(self.gate(concat_emb))

        # Compute news relevance score (r_hat) as a weighted sum
        r_hat = theta * rel_hat_ic + (1 - theta) * rel_hat_nctx

        # Final relevance score taking into consideration user engagement
        relevance_score = ctr * self.w_ctr + r_hat.squeeze(-1) * self.w_r_hat

        return self.sigmoid(relevance_score)


class CompetingAwarePopularityPredictor(nn.Module):
    """Competing Aware Popularity Predictor"""

    def __init__(
        self,
        text_emb_dim: int,
        text_num_heads: int,
        time_emb_dim: int,
        usr_eng_emb_dim: int,
        usr_eng_num_emb: int,
        matrix_size: int,
        dropout_probability: int
    ) -> None:
        super(CompetingAwarePopularityPredictor, self).__init__()

        # Avoidance Aware Relevance Predictor
        self.avw_rel_pred = AvoidanceAwareRelevancePredictor(
            text_emb_dim=text_emb_dim,
            time_emb_dim=time_emb_dim,
        )

        # News multihead attention
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=text_emb_dim, 
            num_heads=text_num_heads,
            dropout=dropout_probability
        )

        # User Engagement Layer
        self.user_eng_layer = nn.Embedding(usr_eng_num_emb, usr_eng_emb_dim)
        self.matrix_size = matrix_size

        # Relevance score Layer
        self.rel_transform = nn.Linear(1, usr_eng_emb_dim)

        self.feature_attention = FeatureAttention(text_emb_dim + 2*usr_eng_emb_dim, 128)

        self.weighted_pooling = WeightedSumPooling(text_emb_dim + 2*usr_eng_emb_dim)

        self.pop_predictor = PopularityPredictor(text_emb_dim + 2*usr_eng_emb_dim)

    def forward(
        self,
        cand_news_vector: Dict[str, torch.Tensor],
        time_elapsed: torch.Tensor,
        ctr: torch.Tensor,
        av_idx: torch.Tensor,
        epi_idx: torch.Tensor,
    ):
        # Compute relevance scores
        relevance_score = self.avw_rel_pred(cand_news_vector, time_elapsed, ctr)
        rs_emb = self.rel_transform(relevance_score.unsqueeze(-1))

        # Compute user engagement embedding 
        user_eng_indices = epi_idx * self.matrix_size + av_idx
        user_eng_emb = self.user_eng_layer(user_eng_indices)

        # Multihead attention
        news_multih_att_emb, _ = self.multihead_attention(cand_news_vector, cand_news_vector, cand_news_vector)

        combined_emb = torch.cat([news_multih_att_emb, user_eng_emb, rs_emb], dim=-1)

        attention_output = self.feature_attention(combined_emb)

        popularity_scores = self.weighted_pooling(attention_output)

        # Get popularity scores
        final_scores = self.pop_predictor(popularity_scores)

        return final_scores
