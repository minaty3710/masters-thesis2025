import numpy as np
from gurobipy import *
import pandas as pd
from scipy.stats import chi2
from gurobipy import Model, GRB, quicksum, QuadExpr

# データの読み込み
file_path = '#RB_weekday.csv'
data = pd.read_csv(file_path)

# データの抽出（列名を統一）
data = data.rename(columns={"week_train": "week", "kakutei_train": "kakutei", "suisyou_train": "suisyou"})

# 14日分のデータをランダムに取得する関数（効率化）
def extract_14_days(data, days_to_sample=7):
    valid_indices = np.arange(len(data) - days_to_sample + 1)
    valid_indices = [i for i in valid_indices if not (
        data['kakutei'][i:i + days_to_sample].isnull().any() or
        data['suisyou'][i:i + days_to_sample].isnull().any()
    )]

    if not valid_indices:
        raise ValueError("条件を満たす14日分のデータが見つかりません。")

    start_idx = np.random.choice(valid_indices)
    kakutei_chunk = data['kakutei'].iloc[start_idx:start_idx + days_to_sample].values
    suisyou_chunk = data['suisyou'].iloc[start_idx:start_idx + days_to_sample].values

    return kakutei_chunk, suisyou_chunk

# 14日分のデータを取得
kakutei_test, suisyou_test = extract_14_days(data)

# パラメータ設定
n_scenarios = 10  # サンプリングするシナリオ数
h = 200  # カーネルのバンド幅

# Nadaraya-Watson法に基づくシナリオ生成関数
def create_demand_scenarios(forecast_demand, d_hat_tau, d_tau, n_scenarios=10, h=200, exclude_indices=None):
    weights = np.exp(-((d_hat_tau - forecast_demand) ** 2) / (2 * h ** 2))

    if np.sum(weights) == 0 or np.isnan(np.sum(weights)):
        weights = np.ones(len(d_hat_tau))
    weights /= weights.sum()

    if exclude_indices:
        valid_indices = [idx for idx in exclude_indices if idx < len(weights)]
        weights[valid_indices] = 0
        weights /= weights.sum()

    selected_indices = np.random.choice(len(d_tau), size=n_scenarios, p=weights, replace=True)
    demand_scenarios = d_tau[selected_indices]
    return demand_scenarios, selected_indices

"需要シナリオを生成"
scenarios = []
used_indices = set()
while len(scenarios) < n_scenarios:
    sampled_demands, sampled_indices = create_demand_scenarios(
        forecast_demand=suisyou_test[0],
        d_hat_tau=data['suisyou'].values,
        d_tau=data['kakutei'].values,
        n_scenarios=n_scenarios - len(scenarios),
        h=h,
        exclude_indices=used_indices
    )

    for global_index in sampled_indices:
        if global_index in used_indices:
            continue
        if global_index + len(suisyou_test) <= len(data['kakutei']):
            scenario = data['kakutei'].iloc[global_index:global_index + len(suisyou_test)].values
            scenarios.append(scenario)
            used_indices.update(range(global_index, global_index + len(suisyou_test)))
        if len(scenarios) >= n_scenarios:
            break

# 必ずnumpy配列に変換
scenarios = np.array(scenarios)  # リストをnumpy配列に変換

# シナリオをデータフレームに変換
scenarios_df = pd.DataFrame(scenarios, columns=[f'Day {i+1}' for i in range(len(suisyou_test))])

# シナリオの結果を表示
print("Generated Demand Scenarios:")
print(scenarios_df)

# サンプリングされたデータから最小体積楕円を求める関数
def find_minimum_volume_ellipsoid(scenarios):
    n_samples, n_dimensions = scenarios.shape

    # Gurobiモデルの作成
    model = Model("MinimumVolumeEllipsoid")

    # 変数の定義
    P = model.addVars(n_dimensions, n_dimensions, name="P")
    rho = model.addVars(n_dimensions, lb=-GRB.INFINITY, name="rho")

    # 対称性制約
    for i in range(n_dimensions):
        for j in range(i+1, n_dimensions):
            model.addConstr(P[i, j] == P[j, i], name=f"Symmetry_P_{i}_{j}")

    # 各点が楕円内に収まる制約
    for idx in range(n_samples):
        sample = scenarios[idx]
        quad_expr = QuadExpr()
        for i in range(n_dimensions):
            quad_expr += (sum(P[i, j] * sample[j] for j in range(n_dimensions)) + rho[i]) ** 2
        model.addQConstr(quad_expr <= 1, name=f"Ellipsoid_Constraint_{idx}")

    # 行列Pの行列式の対数を最小化する目的関数
    model.setObjective(quicksum(P[i, i] for i in range(n_dimensions)), GRB.MINIMIZE)

    # 最適化の実行
    model.optimize()

    # 最適解の取得
    if model.Status == GRB.OPTIMAL:
        P_matrix = np.array([[P[i, j].X for j in range(n_dimensions)] for i in range(n_dimensions)])
        rho_vector = np.array([rho[i].X for i in range(n_dimensions)])

        # 行列R（Pの逆行列）と楕円の中心ūを計算
        R_matrix = np.linalg.inv(P_matrix)
        u_bar = -np.dot(R_matrix, rho_vector)

        return u_bar, R_matrix
    else:
        raise ValueError("Optimal solution not found.")

# 最小体積楕円を求める
u_bar, R_matrix = find_minimum_volume_ellipsoid(scenarios)

# 結果を表示
print("Center of the Ellipsoid (ū):")
print(u_bar)
print("\nShape Matrix of the Ellipsoid (R):")
print(R_matrix)

"データ駆動型ロバスト最適化"

# 時間の数（T期）
T = len(suisyou_test)  # すでに定義されているdays_to_sampleを利用

# 平均需要 (d_bar)
# 先に計算済みのd_barを利用
d_bar = u_bar

# 予測需要 d
# d_testからdays_to_sample分持ってくる
d = suisyou_test[:T]

# 確定需要 d_fix
# d_hat_testからdays_to_sample分持ってくる
d_fix = kakutei_test[:T]

# R行列の調整（悲観的になりすぎないように）
R_matrix = R_matrix*0.05

# パラメータ
h_t = 3  # 保持コスト
b_t = 100  # バックオーダーコスト
M_t = 25000  # 生産量の上限

# モデルの初期化
model = Model("AdaptiveRobustOptimization")

# 変数定義
z = model.addVars(T, T, lb=0, vtype=GRB.CONTINUOUS, name="z")  # アフィン係数
z0 = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="z0")  # アフィン定数
x = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="x")  # 生産量
v = model.addVars(T, T, lb=0, vtype=GRB.CONTINUOUS, name="v")  # v_t
w = model.addVars(T, T, lb=0, vtype=GRB.CONTINUOUS, name="w")  # w_t
y = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="y")  # コスト
norm_v = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="norm_v")  # 保持コスト補助変数
norm_w = model.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="norm_w")  # バックオーダーコスト補助変数

# 目的関数
model.setObjective(quicksum(y[t] for t in range(T)), GRB.MINIMIZE)

# 制約条件
# アフィン関数の制約
for t in range(T):
    if t == 0:  # t = 1
        model.addConstr(x[t] == z0[t], name=f"Affine_x[{t+1}]")
    else:  # t >= 2
        model.addConstr(x[t] == z0[t] + quicksum(z[t, u] * d_fix[u] for u in range(t)), name=f"Affine_x[{t+1}]")

# v_t の定義
for t in range(T):
    for u in range(T):
        if t == 0:  # t = 1
            model.addConstr(v[t, u] == 0, name=f"v[{t+1},{u+1}]")
        elif u < t:  # t >= 2 かつ u < t
            model.addConstr(v[t, u] == z[t, u], name=f"v[{t+1},{u+1}]")
        else:  # u >= t の場合
            model.addConstr(v[t, u] == 0, name=f"v[{t+1},{u+1}]")

# w_t の定義
for t in range(T):
    for u in range(T):
        model.addConstr(w[t, u] == quicksum((1 if s == u else 0) - v[s, u] for s in range(t + 1)), name=f"w[{t},{u}]")

# 制約条件
for t in range(T):
    model.addQConstr(norm_v[t] ** 2 >= quicksum((R_matrix[i, t] * v[t, i]) ** 2 for i in range(T)), name=f"Norm_v[{t}]")
    model.addQConstr(norm_w[t] ** 2 >= quicksum((R_matrix[i, t] * w[t, i]) ** 2 for i in range(T)), name=f"Norm_w[{t}]")

    model.addQConstr(
        y[t] >= h_t * (quicksum(z0[u] for u in range(t + 1)) - quicksum(d_bar[i] * w[t, i] for i in range(T)) + norm_w[t]),
        name=f"HoldingCost[{t}]"
    )
    model.addQConstr(
        y[t] >= b_t * (quicksum(d_bar[i] * w[t, i] for i in range(T)) - quicksum(z0[u] for u in range(t + 1)) + norm_w[t]),
        name=f"BackOrderCost[{t}]"
    )

    model.addQConstr(
        quicksum(d_bar[i] * v[t, i] for i in range(T)) - norm_v[t] + z0[t] >= 0,
        name=f"ProductionMin[{t}]"
    )
    model.addQConstr(
        quicksum(d_bar[i] * v[t, i] for i in range(T)) + norm_v[t] + z0[t] <= M_t,
        name=f"ProductionMax[{t}]"
    )

# モデルの最適化
model.optimize()

# 結果の表示
if model.status == GRB.OPTIMAL:
    print("\n最適解が見つかりました\n")

    # アフィン定数 z0 の表示
    z0_values = [round(z0[t].x, 3) for t in range(T)]
    print("\nアフィン定数 z0:")
    print(z0_values)
    
    # アフィン係数行列 z の表示
    z_matrix = np.array([[round(z[t, u].x, 3) if u <= t else 0 for u in range(T)] for t in range(T)])
    print("\nアフィン係数行列 z:")
    print(z_matrix)
    
    # w 行列の表示
    w_matrix = np.array([[round(w[t, u].x, 3) for u in range(T)] for t in range(T)])
    print("\nw 行列:")
    print(w_matrix)
    
    # v 行列の表示
    v_matrix = np.array([[round(v[t, u].x, 3) for u in range(T)] for t in range(T)])
    print("\nv 行列:")
    print(v_matrix)

    # norm_v の表示
    norm_v_values = [round(norm_v[t].x, 3) for t in range(T)]
    print("\nNorm_v:")
    print(norm_v_values)

    # norm_w の表示
    norm_w_values = [round(norm_w[t].x, 3) for t in range(T)]
    print("\nNorm_w:")
    print(norm_w_values)

    # 実績差の h_cost と b_cost の分解
    h_cost_actual = [int(max(h_t * (d[t] - d_fix[t]), 0)) for t in range(T)]  # h_cost for actual diff
    b_cost_actual = [int(max(b_t * (d_fix[t] - d[t]), 0)) for t in range(T)]  # b_cost for actual diff
    
    # 理論差の h_cost と b_cost の分解
    h_cost_theoretical = [int(max(h_t * (round(x[t].x) - d[t]), 0)) for t in range(T)]  # h_cost for theoretical diff
    b_cost_theoretical = [int(max(b_t * (d[t] - round(x[t].x)), 0)) for t in range(T)]  # b_cost for theoretical diff

    # 保持コスト y_t の表示
    y_values = [int(round(y[t].x)) for t in range(T)]
    print("\n保持コスト y_t:")
    print(y_values)

    # 平均需要 d_bar の表示 (整数)
    print("\n平均需要 (d_bar):")
    print(d_bar)

    # 表形式で表示
    results = pd.DataFrame({
        "期": range(1, T + 1),
        "d_bar": d_bar,
        "d_h_te": d_fix.astype(int),
        "d_te": d.astype(int),
        "x_t": [int(round(x[t].x)) for t in range(T)],
        "y_t": y_values
    })
    
    # 表示フォーマットの調整
    pd.set_option("display.unicode.east_asian_width", True)  # 日本語対応の列幅調整
    pd.set_option("display.colheader_justify", "center")  # 列ヘッダーを中央揃え

    print("\n需要データと生産量:")
    print(results.to_string(index=False))  # インデックスを非表示で列を揃えて表示

    # 累積需要と累積生産量の計算
    cumulative_demand = np.cumsum(d.astype(float))  # 累積需要
    cumulative_production = np.cumsum([round(x[t].x, 3) for t in range(T)])  # 累積生産量
    
    # 累積需要と累積生産量の計算
    cumulative_demand = np.cumsum(d.astype(float))  # 確定需要 d の累積
    cumulative_demand_fix = np.cumsum(d_fix.astype(float))  # 予測需要 d_fix の累積
    cumulative_production = np.cumsum([round(x[t].x, 3) for t in range(T)])  # 生産量 x の累積
    
    # 理論差の h_cost と b_cost の分解 (累積を使用)
    h_cost_theoretical_cumulative = [
        int(max(h_t * (cumulative_production[t] - cumulative_demand[t]), 0)) for t in range(T)
    ]
    b_cost_theoretical_cumulative = [
        int(max(b_t * (cumulative_demand[t] - cumulative_production[t]), 0)) for t in range(T)
    ]
    
    print("\n累積を考慮した理論差の h_cost:")
    print(h_cost_theoretical_cumulative)
    print("\n累積を考慮した理論差の b_cost:")
    print(b_cost_theoretical_cumulative)
    
    # 実績差の h_cost と b_cost の分解 (累積を使用, d_fix と d を基に計算)
    h_cost_actual_cumulative = [
        int(max(h_t * (cumulative_demand_fix[t] - cumulative_demand[t]), 0)) for t in range(T)
    ]
    b_cost_actual_cumulative = [
        int(max(b_t * (cumulative_demand[t] - cumulative_demand_fix[t]), 0)) for t in range(T)
    ]
    
    print("\n累積を考慮した実績差の h_cost:")
    print(h_cost_actual_cumulative)
    print("\n累積を考慮した実績差の b_cost:")
    print(b_cost_actual_cumulative)
    
    # 累積データを含む表形式の出力
    cumulative_results = pd.DataFrame({
        "期": range(1, T + 1),
        "累積予測需要": cumulative_demand,
        "累積実需要": cumulative_demand_fix,
        "累積生産量": cumulative_production,
        "累積理論h": h_cost_theoretical_cumulative,
        "累積理論b": b_cost_theoretical_cumulative,
        "累積実績h": h_cost_actual_cumulative,
        "累積実績b": b_cost_actual_cumulative
    })
    
    print("\n累積需要と累積生産量:")
    print(cumulative_results.to_string(index=False))
    
    # 各累積コストの合計値の計算と表示
    total_h_theoretical_cumulative = sum(h_cost_theoretical_cumulative)
    total_b_theoretical_cumulative = sum(b_cost_theoretical_cumulative)
    total_h_actual_cumulative = sum(h_cost_actual_cumulative)
    total_b_actual_cumulative = sum(b_cost_actual_cumulative)
    
    print("\n累積を考慮した合計値:")
    print(f"累積理論h の合計: {total_h_theoretical_cumulative}")
    print(f"累積理論b の合計: {total_b_theoretical_cumulative}")
    print(f"累積実績h の合計: {total_h_actual_cumulative}")
    print(f"累積実績b の合計: {total_b_actual_cumulative}")
    
    # 累積実績値と累積理論値の総和を表示
    total_theoretical_cumulative = total_h_theoretical_cumulative + total_b_theoretical_cumulative
    total_actual_cumulative = total_h_actual_cumulative + total_b_actual_cumulative
    
    print(f"\n累積理論値の総コスト: {total_theoretical_cumulative}")
    
else:
    print("最適解が見つかりませんでした")

