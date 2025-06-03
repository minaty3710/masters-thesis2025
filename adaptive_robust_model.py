import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import cvxpy as cp

from datetime import datetime
from gurobipy import Model, GRB, quicksum

def generate_sample_dataset(data):
    # 月曜日始まりの 7日間の連続需要データを取得する
    sample_weeks = 7
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
def adaptive_model(df_input,R, d_bar):   
    # Gurobi モデル
    model = Model()
    
    #パラメータ定義
    D = df_input['demand'].tolist()                                            # 期𝑡の需要量 
    #T = len(D)                                                                # 全期間
    T = 7
    Imax = 1500                                                                # 店舗の在庫上限
    Qmax = 500                                                                 # 配送容量上限
    pi = 1000                                                                  # 一日あたりの配送単価
    h = 5                                                                      # 在庫単価
    b = 10                                                                     # 欠品単価
    
    # 決定変数の定義
    q = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="q")                 # 期𝑡の発注量
    y = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="y")                 # 期𝑡の在庫コスト（欠品コスト）  
    delta = model.addVars(T, vtype=GRB.BINARY, name="delta")                   # 配送有無（1のとき配送を実施）
    sigma = model.addVars(T, vtype=GRB.BINARY, name="sigma")                   # 各曜日の配送有無（1の曜日は配送可能）
    
    #補助変数の定義  
    z0 = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="z0") 
    z = model.addVars(T, T, vtype=GRB.CONTINUOUS, lb=0, name="z") 
    v = model.addVars(T, T, vtype=GRB.CONTINUOUS, lb=0, name="v") 
    w = model.addVars(T, T, vtype=GRB.CONTINUOUS, lb=0, name="w")
    norm_Rv= model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="norm_Rv")
    norm_Rw= model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="norm_Rw")
        
    model.update() 

    # 期ごとの曜日のindexを求める関数
    def day_of(u):
        return df_input["day_index"].iloc[u]
    
    # アフィン関数
    for t in range(T):
        if t == 0: 
            model.addConstr(q[t] == z0[t])
        else:  
            model.addConstr(q[t] == z0[t] + quicksum(z[t, u] * D[u] for u in range(t)))

    # 補助変数
    for t in range(T):
        for u in range(T):
            if t == 0:
                model.addConstr(v[t, u] == 0)
            elif u < t:
                model.addConstr(v[t, u] == z[t, u])
            else:
                model.addConstr(v[t, u] == 0)

    for t in range(T):
        for u in range(T):
            model.addConstr(w[t, u] == quicksum((1 if s == u else 0) - v[s, u] for s in range(t + 1)))

    # 制約条件
    for t in range(T):
        model.addQConstr(norm_Rv[t] ** 2 >= quicksum((R[u, t] * v[t, u]) ** 2 for u in range(T)))
        model.addQConstr(norm_Rw[t] ** 2 >= quicksum((R[u, t] * w[t, u]) ** 2 for u in range(T)))

        model.addConstr(y[t] >= h * (quicksum(z0[s] for s in range(t + 1)) - quicksum(d_bar[u] * w[t, u] for  u in range(T)) + norm_Rw[t]))
        model.addConstr(y[t] >= b * (quicksum(d_bar[u] * w[t, u] for  u in range(T)) - quicksum(z0[s] for s in range(t + 1)) + norm_Rw[t]))            
        model.addConstr(quicksum(z0[s] for s in range(t + 1)) - quicksum(d_bar[u] * w[t, u] for  u in range(T)) + norm_Rw[t] <= Imax)
        model.addConstr(z0[t] + quicksum(d_bar[u] * v[t, u] for  u in range(T)) - norm_Rv[t] >= 0)
        model.addConstr(z0[t] + quicksum(d_bar[u] * v[t, u] for  u in range(T)) + norm_Rv[t] <= delta[t] * Qmax) 
        model.addConstr(delta[t] == sigma[day_of(t)])

    # 目的関数
    model.setObjective(quicksum(y[t] + pi * delta[t] for t in range(T)), GRB.MINIMIZE) 
    model.optimize() 
    
    # 結果の出力
    z0_values = [z0[t].X for t in range(T)]
    q_values = [q[t].X for t in range(T)] 
    y_values = [y[t].X for t in range(T)] 
    delta_values = [delta[t].X for t in range(T)] 
    sigma_values = [sigma[df_input["day_index"].iloc[t]].X for t in range(T)] 
    inventory = [
        max(0, sum(z0_values[s] for s in range(t + 1)) - sum(D[i] * w[t, i].X for i in range(T)))
        for t in range(T)
    ]
    out_of_stock = [
        max(0, sum(D[i] * w[t, i].X for i in range(T)) - sum(z0_values[s] for s in range(t + 1)))
        for t in range(T)
    ]
    delivery_costs = [pi * delta[t].X for t in range(T)] 
    inventory_costs = [h * inventory[t] for t in range(T)] 
    out_of_stock_costs = [b * out_of_stock[t] for t in range(T)] 
    v_values = [[v[t, u].X for u in range(T)] for t in range(T)]
    w_values = [[w[t, u].X for u in range(T)] for t in range(T)] 
    z_values = [[z[t, u].X for u in range(T)] for t in range(T)] 

    # 結果をデータフレームとして格納
    date_list = df_input['date'].tolist()
    weekday_list = df_input['date'].dt.strftime('%a').tolist()       
    df_results = pd.DataFrame({
        'Date': date_list[:T],  
        'week_day': weekday_list[:T],
        'Demand': D[:T],
        'Order Quantity': q_values,
        'y_Cost' : y_values,
        'Inventory': inventory,
        'out_of_stock': out_of_stock,
        'delta': delta_values,
        'sigma' : sigma_values,
        'Delivery Cost': delivery_costs,
        'Inventory Cost': inventory_costs,
        'out_of_stock Cost': out_of_stock_costs,
        'v_values': [str(v_values[t]) for t in range(T)],  
        'w_values': [str(w_values[t]) for t in range(T)],
        'z0_values': z0_values,
        'z_values': [str(z_values[t]) for t in range(T)],
    })

    return  df_results 

def plot_order_quantity(df_results):
    #リザルトデータの読み込み
    date_list = df_results['Date'].tolist() 
    x_values = df_results['Order Quantity'].tolist() 
    plt.figure(figsize=(12, 4))
    plt.plot(date_list, x_values, marker='o', linestyle='-', color='green', label='Order Quantity')
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Order Quantity')
    plt.title('Optimal Order Quantity over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f'order_quantity_{timestamp}.png'
    save_path = os.path.join("C:/Users/mina1/.spyder-py3/master's thesis/result", filename)
    plt.savefig(save_path, dpi=300)
    plt.show()
    
def export_results_to_csv(df_results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f'optimization_result_{timestamp}.csv'
    save_path = os.path.join("C:/Users/mina1/.spyder-py3/master's thesis/result", filename)

    # CSV 出力
    df_results.to_csv(save_path, index=False)
    print(f"結果を保存しました: {save_path}")

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
df_results = adaptive_model(test_data, R, d_bar)

plot_order_quantity(df_results)
export_results_to_csv(df_results)