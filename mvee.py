
import pandas as pd
import numpy as np
import cvxpy as cp

from datetime import datetime
from gurobipy import Model, GRB, quicksum

def generate_sample_dataset(data):
    # 月曜日始まりの 7日間の連続需要データを取得する
    sample_weeks = 5
    seed = None
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
    selected_starts = np.random.choice(valid_starts, size=sample_weeks, replace=False)
    
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



read_path = "C:\\Users\mina1\.spyder-py3\master's thesis\dataset\demand_data_2025-04-22_1416.csv" 
df_input = pd.read_csv(read_path) 
# 日付が datetime 型でなければ変換
df_input["date"] = pd.to_datetime(df_input["date"]) 
# 月曜日を0とした曜日インデックスを列に追加
df_input["day_index"] = df_input["date"].dt.weekday 
# training_data/test_data にもこの列を継承
training_data = df_input[(df_input["date"].dt.year == 2025) & (df_input["date"].dt.month <= 2)].copy() 
test_data = df_input[(df_input["date"].dt.year == 2025) & (df_input["date"].dt.month >= 3)].copy() 
demand_samples  = generate_sample_dataset(training_data)
R, d_bar = minimum_volume_enclosing_ellipsoid(demand_samples)
print("R:\n", R)
print("d̄:\n", d_bar)