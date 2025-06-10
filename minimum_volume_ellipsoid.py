import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import cvxpy as cp

from datetime import datetime

# ---- 楕円体を描く関数 ----
def plot_ellipse(R, d_bar, ax, label, color):
    theta = np.linspace(0, 2 * np.pi, 100)
    unit_circle = np.stack([np.cos(theta), np.sin(theta)])  # shape (2, 100)
    ellipse = d_bar.reshape(2, 1) + R @ unit_circle
    ax.plot(ellipse[0], ellipse[1], label=label, color=color)

def generate_sample_dataset(data, n_samples=10, seed=None):
    # 月曜日始まりの 7日間の連続需要データを取得す
    if seed is not None:
        np.random.seed(seed)
    
    weekly_samples = []
    monday_indices = data.index[data["day_index"] == 0].tolist()
    
    # 各月曜日について、7日分のデータが連続で取れるか確認
    valid_starts = []
    for idx in monday_indices:
        if idx + 6 < len(data):
            # 曜日が連続しているか（日付も連続しているか確認）
            start_date = data.loc[idx, "date"]
            end_date = data.loc[idx + 6, "date"]
            if (end_date - start_date).days == 6:
                valid_starts.append(idx)
    
    # ランダムに週を選ぶ（重複なし）
    selected_starts = np.random.choice(valid_starts, size=n_samples, replace=False)
    
    for start_idx in selected_starts:
        week_demand = data.loc[start_idx:start_idx+6, "demand"].values
        weekly_samples.append(week_demand)

    return np.array(weekly_samples)
def minimum_volume_enclosing_ellipsoid(samples):
    N, T = samples.shape

    # 変数の定義
    P   = cp.Variable((T, T), PSD=True) 
    rho = cp.Variable(T)

    # 制約
    constraints = [cp.norm(P @ samples[i] + rho, 2) <= 1 for i in range(N)]

    # 目的関数
    objective = cp.Minimize(-cp.log_det(P))

    # 最適化
    prob = cp.Problem(objective, constraints)
    prob.solve(solver="SCS")

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Solver failed: {prob.status}")

    R       = np.linalg.inv(P.value)
    d_bar   = - R @ rho.value

    return R, d_bar

read_path = "C:/Users/mina1/.spyder-py3/master's thesis/dataset/demand_data_2025-06-09_1643.csv"
df_input = pd.read_csv(read_path) 
# 日付が datetime 型でなければ変換
df_input["date"] = pd.to_datetime(df_input["date"]) 
# 月曜日を0とした曜日インデックスを列に追加
df_input["day_index"] = df_input["date"].dt.weekday 
# training_data/test_data にもこの列を継承
training_data = df_input[(df_input["date"].dt.year == 2024)].copy() 
test_data = df_input[(df_input["date"].dt.year == 2025)].copy() 

# サンプル数別に楕円体を描画
fig, ax = plt.subplots(figsize=(8, 8))
colors = ['red', 'green', 'blue']

for n_samples, color in zip([10, 20, 30], colors):
    # サンプル抽出（2日間限定）
    demand_samples = generate_sample_dataset(training_data, n_samples=n_samples)
    demand_samples = demand_samples[:, :2]  # 先頭2日間に限定

    # 楕円体計算
    R, d_bar = minimum_volume_enclosing_ellipsoid(demand_samples)

    # サンプル点の描画
    ax.scatter(demand_samples[:, 0], demand_samples[:, 1], alpha=0.4, label=f"Samples {n_samples}", color=color)
    
    # 楕円体描画
    plot_ellipse(R, d_bar, ax, label=f"MVEE (n={n_samples})", color=color)

# 軸と凡例
ax.set_xlabel("Monday")
ax.set_ylabel("Tuesday")
ax.set_title("MVEE")
ax.legend()
ax.grid(True)
plt.axis('equal')
plt.tight_layout()
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
filename = f'MVEE_{timestamp}.png'
save_path = os.path.join("C:/Users/mina1/.spyder-py3/master's thesis/result", filename)
plt.savefig(save_path, dpi=300)
plt.show()