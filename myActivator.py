import torch
import torch.nn as nn

class tanhAndScale(nn.Module):
    def __init__(self, a_low, a_high, dtype=torch.float32):
        super().__init__()

        # a_low / a_high は list, numpy, torch.Tensor のいずれでも受け取れるようにする
        a_low = torch.as_tensor(a_low, dtype=dtype)
        a_high = torch.as_tensor(a_high, dtype=dtype)

        # 形状チェック（次元ごと制約なら通常は (act_dim,)）
        # スカラーでもブロードキャストで動くので許可するが、low/high の形状は一致させる
        if a_low.shape != a_high.shape:
            raise ValueError(f"a_low.shape={a_low.shape} and a_high.shape={a_high.shape} must match")
        
        # 範囲チェック（low < high を仮定）
        # 全次元で low < high になっているか確認
        if not torch.all(a_low < a_high):
            raise ValueError("All elements must satisfy a_low < a_high")
        
        # 変換を a = mid + scale * tanh(u) で書くための定数を作る
        mid = (a_high + a_low) * 0.5
        scale = (a_high - a_low) * 0.5

        # これらは学習パラメータではなく定数なので buffer 登録する
        # model.to(device) で一緒にGPUへ移動し、state_dictにも入る
        self.register_buffer("mid", mid)
        self.register_buffer("scale", scale)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u の最後の次元が act_dim と一致することを仮定（ベクトル境界の場合）
        # mid/scale がスカラーならこのチェックは不要だが、ベクトル時の事故防止に入れている
        if self.mid.ndim == 1 and u.shape[-1] != self.mid.shape[0]:
            raise ValueError(f"Last dim of u is {u.shape[-1]}, but act_dim is {self.mid.shape[0]}")

        # 1) tanh で (-1,1) に押し込む（要素ごと）
        z = torch.tanh(u)

        # 2) 次元ごとの mid/scale をブロードキャストして変換
        #    a = mid + scale * z
        return self.mid + self.scale * z