import torch
from torch.distributions import Normal, Independent, TransformedDistribution
from torch.distributions.transforms import TanhTransform, AffineTransform

def make_squashed_gaussian(mu: torch.Tensor, std: torch.Tensor,
                          low: torch.Tensor, high: torch.Tensor):
    # 1) ベース分布：多次元の対角ガウス（Independentで次元和を自動化）
    base = Independent(Normal(loc=mu, scale=std), 1)

    # 2) (-1,1) に押し込む tanh 変換（可逆変換として扱える）
    tanh = TanhTransform(cache_size=1)

    # 3) (-1,1) -> (low, high) にアフィン変換
    #    a = scale * x + loc,  scale = (high-low)/2, loc = (high+low)/2
    scale = (high - low) / 2.0
    loc = (high + low) / 2.0
    affine = AffineTransform(loc=loc, scale=scale)

    # 4) 変換付き分布：log_prob はヤコビアン補正込みで計算される
    dist = TransformedDistribution(base, [tanh, affine])
    return dist

# ---- 使い方（例）----
# mu: [B, act_dim], std: [B, act_dim]
# low/high: [act_dim] でも [B, act_dim] でもOK（ブロードキャストされる）
# dist = make_squashed_gaussian(mu, std, low, high)
# a = dist.rsample()          # a: [B, act_dim] かつ必ず範囲内
# logp = dist.log_prob(a)     # logp: [B]（多次元和も含めて1値）
