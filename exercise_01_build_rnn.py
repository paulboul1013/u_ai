"""練習 1：僅使用 NumPy，從零實作字元層級 RNN。

資料流：字元索引 → one-hot → 隱藏狀態 → 下一字元機率。
訓練流程：forward → backward（BPTT 與梯度裁剪）→ update_weights。
執行：python3 exercises/exercise_01_build_rnn.py
"""

import numpy as np


class CharRNN:
    """單層 tanh RNN，使用交叉熵損失與隨機梯度下降。

    所有向量都是欄向量。V 表示字元種類數，H 表示隱藏狀態維度。
    同一組參數會重複用在每個時間步。
    """

    def __init__(self, vocab_size, hidden_size, seq_length, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate

        # 維度由 W @ 欄向量決定；小隨機值讓各隱藏單元有不同起點。
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # H × V
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # H × H
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01  # V × H
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))

    def forward(self, inputs, targets, h_prev):
        """讀取 inputs，計算預測 targets 的損失。

        inputs、targets 必須等長且非空。targets 取自原始資料的
        下一個位置，例如輸入 abc，目標 bca。
        回傳整段損失總和、最後的隱藏狀態，以及 BPTT 所需的 cache。
        """
        if len(inputs) == 0 or len(inputs) != len(targets):
            raise ValueError("inputs and targets must be non-empty and have equal length")

        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        loss = 0.0

        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t], 0] = 1
            hs[t] = np.tanh(self.Wxh @ xs[t] + self.Whh @ hs[t - 1] + self.bh)
            ys[t] = self.Why @ hs[t] + self.by

            # 平移 logits，避免 exp 溢位；平移不改變 softmax 機率。
            shifted = ys[t] - np.max(ys[t])
            log_probs = shifted - np.log(np.sum(np.exp(shifted)))
            ps[t] = np.exp(log_probs)
            # 直接用 log 機率，避免先算出 0 再取 log，得到無限大。
            loss -= log_probs[targets[t], 0]

        return float(loss), hs[len(inputs) - 1], (xs, hs, ys, ps)

    def backward(self, inputs, targets, cache):
        """沿時間倒序套用鏈式法則，累加共享參數的梯度，再裁剪。"""
        xs, hs, ys, ps = cache
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros_like(hs[-1])

        for t in reversed(range(len(inputs))):
            # softmax + 交叉熵的導數：預測機率減去正確答案的 one-hot。
            dy = ps[t].copy()
            dy[targets[t], 0] -= 1
            dWhy += dy @ hs[t].T
            dby += dy

            # h_t 同時影響當下輸出與未來時間步，兩條路徑的梯度要相加。
            dh = self.Why.T @ dy + dh_next
            dh_raw = (1 - hs[t] ** 2) * dh
            dWxh += dh_raw @ xs[t].T
            dWhh += dh_raw @ hs[t - 1].T
            dbh += dh_raw
            dh_next = self.Whh.T @ dh_raw

        # 全部時間步累加後才裁剪，限制每個梯度元素在 [-5, 5]。
        for gradient in (dWxh, dWhh, dWhy, dbh, dby):
            np.clip(gradient, -5, 5, out=gradient)

        return {'Wxh': dWxh, 'Whh': dWhh, 'Why': dWhy, 'bh': dbh, 'by': dby}

    def update_weights(self, grads):
        """梯度指向損失增加的方向，因此沿反方向更新參數。"""
        for name, gradient in grads.items():
            parameter = getattr(self, name)
            parameter -= self.learning_rate * gradient

    def sample(self, h, seed_idx, n):
        """從 seed 字元開始，將每次取樣結果當作下一步輸入。

        回傳 n 個生成字元的索引，不包含 seed；此方法不更新模型參數。
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_idx, 0] = 1
        indices = []

        for _ in range(n):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y = self.Why @ h + self.by
            exp_y = np.exp(y - np.max(y))
            p = exp_y / np.sum(exp_y)
            idx = int(np.random.choice(self.vocab_size, p=p.ravel()))
            indices.append(idx)
            x.fill(0)
            x[idx, 0] = 1

        return indices


def train_rnn(iterations=1000):
    """用 abc 重複序列示範訓練；一次 iteration 更新一個序列片段。"""
    data = "abcabcabcabcabcabc"
    chars = sorted(set(data))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    data_indices = [char_to_idx[ch] for ch in data]
    rnn = CharRNN(vocab_size=len(chars), hidden_size=10, seq_length=3,
                  learning_rate=0.1)

    h_prev = np.zeros((rnn.hidden_size, 1))
    position = 0
    for iteration in range(iterations):
        # 需要額外一字元當最後的 target；回到資料開頭時重設記憶。
        if position + rnn.seq_length >= len(data_indices):
            position = 0
            h_prev = np.zeros_like(h_prev)

        inputs = data_indices[position:position + rnn.seq_length]
        targets = data_indices[position + 1:position + rnn.seq_length + 1]
        loss, h_prev, cache = rnn.forward(inputs, targets, h_prev)
        grads = rnn.backward(inputs, targets, cache)
        rnn.update_weights(grads)
        position += rnn.seq_length

        if iteration % 100 == 0 or iteration == iterations - 1:
            print(f"Iteration {iteration:4d} | loss/char = {loss / len(inputs):.6f}")

    print("Training complete!")
    h = np.zeros((rnn.hidden_size, 1))
    indices = rnn.sample(h, seed_idx=char_to_idx['a'], n=30)
    text = ''.join(idx_to_char[i] for i in indices)
    print(f"Seed: a | Generated: {text}")
    return rnn


if __name__ == "__main__":
    np.random.seed(42)
    train_rnn()
