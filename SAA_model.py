import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import cvxpy as cp

from datetime import datetime,timedelta
from gurobipy import Model, GRB, quicksum

def generate_sample_dataset(data):
    # 月曜日始まりの 7日間の連続需要データを取得する
    sample_weeks = 10
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
def convert_samples_to_dataframe(demand_samples, reference_data):
    result = []

    # ランダムな月曜の日付をサンプルと同じ数だけ取得
    base_dates = reference_data[reference_data["day_index"] == 0]["date"].reset_index(drop=True)
    base_dates = base_dates.sample(n=demand_samples.shape[0], replace=False).reset_index(drop=True)

    for week_idx, week in enumerate(demand_samples):
        base_date = base_dates[week_idx]
        for day_offset in range(7):
            date = base_date + timedelta(days=day_offset)
            demand = week[day_offset]
            day_index = (base_date.weekday() + day_offset) % 7

            result.append({
                "date": date,
                "demand": demand,
                "day_index": day_index
            })

    df = pd.DataFrame(result)
    return df
def pre_sample_average_approximation_model(df_input):
    # Gurobi モデル
    model = Model()
    
    #パラメータ定義
    D = df_input['demand'].tolist()                                            # 期𝑡の需要量 
    T = len(D)                                                                 # 全期間
    Imax = 1500                                                                # 店舗の在庫上限
    Qmax = 500                                                                 # 配送容量上限
    pi = 1000                                                                  # 一日あたりの配送単価
    h = 5                                                                      # 在庫単価
    b = 20                                                                     # 欠品単価
    
    # 決定変数の定義
    q = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="q")                 # 期𝑡の発注量
    y = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="y")                 # 期𝑡の在庫コスト（欠品コスト）  
    delta = model.addVars(T, vtype=GRB.BINARY, name="delta")                   # 配送有無（1のとき配送を実施）
    sigma = model.addVars(T, vtype=GRB.BINARY, name="sigma")                   # 各曜日の配送有無（1の曜日は配送可能）
        
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
        i = day_of(t)         
        model.addQConstr(norm_Rv[t] ** 2 >= quicksum((R[i, day_of(u)] * v[t, u]) ** 2 for u in range(T)))
        model.addQConstr(norm_Rw[t] ** 2 >= quicksum((R[i, day_of(u)] * w[t, u]) ** 2 for u in range(T)))
        # 累積需要・供給
        demand_sum = quicksum(d_bar[day_of(u)] * w[t, u] for u in range(T))
        supply_sum = quicksum(z0[s] for s in range(t + 1))
        # 在庫コスト/欠品コスト
        model.addConstr(y[t] >= h * (supply_sum - demand_sum + norm_Rw[t]))
        model.addConstr(y[t] >= b * (demand_sum - supply_sum + norm_Rw[t]))
        # (2.d)店舗在庫の容量制約       
        model.addConstr(supply_sum - demand_sum + norm_Rw[t] <= Imax)
        # (2.e)発注量の制約
        model.addConstr(z0[t] + quicksum(d_bar[day_of(u)] * v[t, u] for u in range(T)) - norm_Rv[t] >= 0)
        model.addConstr(z0[t] + quicksum(d_bar[day_of(u)] * v[t, u] for u in range(T)) + norm_Rv[t] <= delta[t] * Qmax)     
        # 曜日の制約
        model.addConstr(delta[t] == sigma[i])

    # 目的関数
    model.setObjective(quicksum(y[t] + pi * delta[t] for t in range(T)), GRB.MINIMIZE) 
    model.optimize() 
    
    # 結果の出力
    delivery_schedule = [int(round(sigma[i].X)) for i in range(7)]
    return delivery_schedule  
def sample_average_approximation_model(df_input,delivery_schedule):
    # Gurobi モデル
    model = Model()
    
    #パラメータ定義
    D = df_input['demand'].tolist()                                            # 期𝑡の需要量 
    T = len(D)                                                                 # 全期間
    Imax = 1500                                                                # 店舗の在庫上限
    Qmax = 500                                                                 # 配送容量上限
    pi = 1000                                                                  # 一日あたりの配送単価
    h = 5                                                                      # 在庫単価
    b = 20                                                                     # 欠品単価
    sigma = delivery_schedule[:7]                                              # 曜日配送スケジュール
    
    # 決定変数の定義
    q = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="q")                 # 期𝑡の発注量
    y = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="y")                 # 期𝑡の在庫コスト（欠品コスト）  
    delta = model.addVars(T, vtype=GRB.BINARY, name="delta")                   # 配送有無（1のとき配送を実施）
    
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
        i = day_of(t)         
        model.addQConstr(norm_Rv[t] ** 2 >= quicksum((R[i, day_of(u)] * v[t, u]) ** 2 for u in range(T)))
        model.addQConstr(norm_Rw[t] ** 2 >= quicksum((R[i, day_of(u)] * w[t, u]) ** 2 for u in range(T)))
        # 累積需要・供給
        demand_sum = quicksum(d_bar[day_of(u)] * w[t, u] for u in range(T))
        supply_sum = quicksum(z0[s] for s in range(t + 1))
        # 在庫コスト/欠品コスト
        model.addConstr(y[t] >= h * (supply_sum - demand_sum + norm_Rw[t]))
        model.addConstr(y[t] >= b * (demand_sum - supply_sum + norm_Rw[t]))
        # (2.d)店舗在庫の容量制約       
        model.addConstr(supply_sum - demand_sum + norm_Rw[t] <= Imax)
        # (2.e)発注量の制約
        model.addConstr(z0[t] + quicksum(d_bar[day_of(u)] * v[t, u] for u in range(T)) - norm_Rv[t] >= 0)
        model.addConstr(z0[t] + quicksum(d_bar[day_of(u)] * v[t, u] for u in range(T)) + norm_Rv[t] <= delta[t] * Qmax)     
        # 曜日の制約
        model.addConstr(delta[t] == sigma[i])

    # 目的関数
    model.setObjective(quicksum(y[t] + pi * delta[t] for t in range(T)), GRB.MINIMIZE) 
    model.optimize() 
    
    # 結果の出力
    z0_values = [z0[t].X for t in range(T)]
    q_values = [q[t].X for t in range(T)] 
    y_values = [y[t].X for t in range(T)] 
    delta_values = [delta[t].X for t in range(T)] 
    sigma_values = [sigma[df_input["day_index"].iloc[t]] for t in range(T)]
    inventory = [
        max(0, sum(z0_values[s] for s in range(t + 1)) - sum(D[i] * w[t, i].X for i in range(T)))
        for t in range(T)
    ]
    out_of_stock = [
        max(0, sum(D[i] * w[t, i].X for i in range(T)) - sum(z0_values[s] for s in range(t + 1)))
        for t in range(T)
    ]
    norm_Rw_values = [norm_Rw[t].X for t in range(T)] 
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
        'norm_Rw_values' : norm_Rw_values,
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
    filename = f'robust_model_result_{timestamp}.png'
    save_path = os.path.join("C:/Users/mina1/.spyder-py3/master's thesis/result", filename)
    plt.savefig(save_path, dpi=300)
    plt.show()  
def export_results_to_csv(df_results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f'robust_model_result_{timestamp}.csv'
    save_path = os.path.join("C:/Users/mina1/.spyder-py3/master's thesis/result", filename)

    # CSV 出力
    df_results.to_csv(save_path, index=False)
    print(f"結果を保存しました: {save_path}")

read_path = "C:/Users/mina1/.spyder-py3/master's thesis/dataset/demand_data_2025-06-09_1643.csv"
df_input = pd.read_csv(read_path) 
# 日付が datetime 型でなければ変換
df_input["date"] = pd.to_datetime(df_input["date"]) 
# 月曜日を0とした曜日インデックスを列に追加
df_input["day_index"] = df_input["date"].dt.weekday 
# training_data/test_data にもこの列を継承
training_data = df_input[(df_input["date"].dt.year == 2024)].copy() 
test_data = df_input[(df_input["date"].dt.year == 2025)].copy() 
demand_samples  = generate_sample_dataset(training_data)
df_samples = convert_samples_to_dataframe(demand_samples, training_data)
delivery_schedule = pre_sample_average_approximation_model(df_samples)
print(delivery_schedule)
df_results = sample_average_approximation_model(test_data, delivery_schedule)

#plot_order_quantity(df_results)
#export_results_to_csv(df_results)