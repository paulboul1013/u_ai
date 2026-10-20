# 從零實作字元 RNN：學習進度與說明

[exercise_01_build_rnn.py](exercise_01_build_rnn.py) 僅使用 NumPy 實作字元 RNN，包含前向計算、BPTT、梯度裁剪、SGD 更新與逐字取樣。
本資料夾只保留這份原始檔與本說明，學習重點及驗證結果集中於此。

## 任務與模型設定

訓練資料是 `abcabcabcabcabcabc`，輸入 `abc`，目標為下一個字元 `bca`。
模型要學會 `a → b`、`b → c`、`c → a`。

| 項目 | 設定 |
|---|---|
| 字元表 | `a=0`、`b=1`、`c=2` |
| 輸入方式 | one-hot 欄向量 |
| 隱藏維度 | 10 |
| 每段序列長度 | 3 |
| 訓練更新次數 | 1000 |
| 學習率與最佳化器 | 0.1，SGD |
| 可訓練參數 | 173 個 |
| 主程式亂數種子 | 42 |

## 程式如何運作

| 函式 | 工作 |
|---|---|
| `CharRNN.__init__()` | 初始化三組權重與兩組偏置 |
| `forward(inputs, targets, h_prev)` | 計算每一步的隱藏狀態、機率與交叉熵，保存反向計算需要的快取 |
| `backward(inputs, targets, cache)` | 沿時間倒序計算梯度，累加共享參數的梯度並裁剪 |
| `update_weights(grads)` | 以 SGD 更新參數 |
| `sample(h, seed_idx, n)` | 依預測機率取樣，將生成字元當作下一步輸入 |
| `train_rnn()` | 切分資料、傳遞隱藏狀態、安排訓練並列印生成結果 |

前向計算使用以下公式：

```text
h_t = tanh(Wxh @ x_t + Whh @ h_(t-1) + bh)
y_t = Why @ h_t + by
p_t = softmax(y_t)
L   = -Σ log(p_t[target_t])
```

`x_t` 的形狀為 `3×1`，`h_t` 為 `10×1`。
`Wxh`、`Whh`、`Why` 的形狀分別是 `10×3`、`10×10`、`3×10`。
同一組權重用於每個時間步，隱藏狀態則隨輸入改變。
程式先平移 logits，再計算 log 機率，避免指數溢位及對零取對數。

BPTT 從最後一個時間步往前計算。
每個隱藏狀態的梯度同時來自當下輸出與後續時間步：

```text
dy[target_t] -= 1          # dy 原先是預測機率 p_t 的副本
dh = Why.T @ dy + dh_next
dh_raw = (1 - h_t²) * dh   # tanh 的導數
dh_next = Whh.T @ dh_raw
```

每一步對同一參數的梯度必須累加，全部累加後才逐元素裁剪至 `[-5, 5]`。
SGD 使用 `parameter -= learning_rate * gradient` 更新參數。
訓練跨片段傳遞隱藏狀態，但梯度只在當前片段內反傳，屬於截斷 BPTT。
回到訓練資料開頭時，隱藏狀態歸零。

生成時沒有提供正確答案，也不更新參數。
從種子字元 `a` 開始取樣，回傳的生成文字不包含種子本身。

## 執行方式

需要 Python 3 與 NumPy。本次驗證環境為 Python 3.12.3、NumPy 2.4.1。

```bash
cd /home/paulboul/u_ai
python3 -m pip install numpy==2.4.1
python3 exercise_01_build_rnn.py
```

## 學習成果與驗證結果

2026-10-20 重新執行本資料夾中的原始檔，輸出如下：

```text
Iteration    0 | loss/char = 1.098666
Iteration  100 | loss/char = 0.031824
Iteration  200 | loss/char = 0.011140
Iteration  300 | loss/char = 0.006596
Iteration  400 | loss/char = 0.004644
Iteration  500 | loss/char = 0.003567
Iteration  600 | loss/char = 0.002888
Iteration  700 | loss/char = 0.002421
Iteration  800 | loss/char = 0.002081
Iteration  900 | loss/char = 0.001824
Iteration  999 | loss/char = 0.000425
Training complete!
Seed: a | Generated: bcabcabcabcabcabcabcabcabcabca
```

每字元損失從 `1.098666` 降到 `0.000425`，模型能生成連續的 `bcabcabc…`。
這說明參數經過更新後，能學會此資料的下一字元規律。

同日精簡前，既有 6 項測試全部通過，涵蓋以下項目：

- 均勻預測時的交叉熵。
- 極端 logits 下的有限損失。
- 拒絕空序列與輸入、目標長度不一致。
- 所有參數的 BPTT 梯度與有限差分數值梯度一致。
- 梯度累加後的裁剪範圍。
- 訓練能降低損失，並正確預測與生成下一字元。

目前兩檔版本可直接重跑訓練。測試檔、比較報告及圖解教學仍保留於原始練習資料夾：
`/home/paulboul/30u30/papers/01_Unreasonable_Effectiveness/exercises/`。

這些成果佐證 RNN 基礎實作及簡單序列任務的學習結果。
`abc` 的下一字元只由目前字元決定，因此這項實驗無法證明長距離記憶或一般自然語言生成能力。
