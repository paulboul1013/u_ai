"""在簡單的線性迴歸問題上比較 SGD 與 Adam。

五筆資料都符合 ``y = 2x + 1``。每個最佳化方法都從 ``w = 0``、``b = 0``
開始，並降低模型 ``y_hat = w * x + b`` 的均方誤差（MSE）。訓練成功時，
``w`` 會接近 2，``b`` 會接近 1。
"""

import math


# 訓練資料格式為（輸入 x, 正確答案 y）。
data = [(-2., -3.), (-1., -1.), (0., 1.), (1., 3.), (2., 5.)]


def loss(w, b):
    """計算參數 ``w`` 與 ``b`` 在全部資料上的均方誤差。"""
    return sum((w * x + b - y) ** 2 for x, y in data) / len(data)


def gradients(w, b):
    """直接用微分公式計算 MSE 梯度 ``(dL/dw, dL/db)``。

    對單筆資料定義 ``error = y_hat - y``。將 ``error**2`` 分別對 ``w``
    與 ``b`` 微分後，會得到 ``2 * error * x`` 與 ``2 * error``。最後再對
    全部資料的梯度取平均。
    """
    errors = [(x, w * x + b - y) for x, y in data]
    return (
        sum(2 * error * x for x, error in errors) / len(data),
        sum(2 * error for _, error in errors) / len(data),
    )


# 從 w=0、b=0 開始，並以很小的有限差分步長 eps 檢查 gradients() 的公式。
# assert 會比較解析梯度與數值梯度。如果誤差過大，程式會立即停止。
w, b, eps = 0., 0., 1e-6
gw, gb = gradients(w, b)
assert abs(gw - (loss(w + eps, b) - loss(w - eps, b)) / (2 * eps)) < 1e-7
assert abs(gb - (loss(w, b + eps) - loss(w, b - eps)) / (2 * eps)) < 1e-7

# 在完整訓練前，先顯示初始損失與梯度。
print('01 初始 MSE:', loss(w, b))
print('02 梯度 dw,db:', gw, gb)

# 以學習率 0.1 執行一次 SGD 更新。更新後損失降低，表示參數往負梯度方向移動。
print('03 SGD 一步:', loss(w - .1 * gw, b - .1 * gb))


class Adam:
    """專門更新 ``w`` 與 ``b`` 的精簡版 Adam 最佳化器。

    ``m`` 保存梯度的指數移動平均，``v`` 保存梯度平方的指數移動平均。
    Adam 根據兩者為每個參數調整更新幅度。
    """

    def __init__(self):
        self.m = [0., 0.]
        self.v = [0., 0.]
        self.t = 0

    def step(self, params, grads, lr):
        """執行一次 Adam 更新，並回傳更新後的參數。"""
        self.t += 1
        result = []
        for i, (parameter, gradient) in enumerate(zip(params, grads)):
            # 使用 beta1=0.9 與 beta2=0.999 更新一階動量和二階動量。
            self.m[i] = .9 * self.m[i] + .1 * gradient
            self.v[i] = .999 * self.v[i] + .001 * gradient * gradient

            # 訓練初期的移動平均會偏向零，因此要除以累積的 beta 次方來修正偏差。
            m_hat = self.m[i] / (1 - .9 ** self.t)
            v_hat = self.v[i] / (1 - .999 ** self.t)

            # 分母加上小常數，避免 v_hat 為零時發生除以零。
            result.append(parameter - lr * m_hat / (math.sqrt(v_hat) + 1e-8))
        return result


def schedule(step, total=300, warmup=30, peak=.1, minimum=.001):
    """計算目前步數的 warmup 加 cosine 學習率。

    warmup 階段將學習率從零線性增加到 ``peak``。剩餘步數使用半個 cosine
    曲線，將學習率從 ``peak`` 平滑降低到 ``minimum``。
    """
    if step <= warmup:
        return peak * step / warmup

    progress = (step - warmup) / (total - warmup)
    return minimum + .5 * (peak - minimum) * (
        1 + math.cos(math.pi * progress)
    )


def train(method, total=300):
    """使用 ``sgd``、``adam`` 或 ``adam_schedule`` 進行訓練。

    每筆 history 紀錄都包含 ``(步數, 學習率, 損失, w, b)``，可用來比較三種
    方法的收斂過程。
    """
    w, b = 0., 0.
    optimizer = Adam()
    history = []

    for step in range(1, total + 1):
        # 只有 adam_schedule 會在訓練過程中調整學習率，其餘方法固定使用 0.1。
        lr = schedule(step, total) if method == 'adam_schedule' else .1
        gw, gb = gradients(w, b)

        if method == 'sgd':
            w, b = w - lr * gw, b - lr * gb
        else:
            w, b = optimizer.step([w, b], [gw, gb], lr)

        history.append((step, lr, loss(w, b), w, b))

    return history


if __name__ == '__main__':
    # 三種方法都從相同初始值開始。理想結果是 w=2、b=1，且損失接近零。
    for method in ['sgd', 'adam', 'adam_schedule']:
        history = train(method)
        print(method, '最終 step,lr,loss,w,b:', history[-1])
