import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from  datetime import datetime
from gurobipy import *

def static_robust_optimization(df_input):   
    # Gurobi モデル
    model = model()
    
    # 決定変数の定義
    q = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="q")                 # 期𝑡の発注量
    y = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="y")                 # 期𝑡の在庫コスト（欠品コスト）  
    delta = model.addVars(T, vtype=GRB.BINARY, name="delta")                   # 配送有無（1のとき配送を実施）
    sigma = model.addVars(T, vtype=GRB.BINARY, name="sigma")                   # 各曜日の配送有無（1の曜日は配送可能）
    
    #パラメータ設定
    T = len(demand_list)                                                       # 全期間
    Upsilon = {t: list(range(t)) for t in range(T)}                            # 𝑡−1期までの集合
    D = df_input['demand'].tolist()                                            # 期𝑡の需要量 
    d_mean = np.mean(D)                                                        # 期𝑡の需要量の平均
    Imax = 1500                                                                # 店舗の在庫上限
    Qmax = 500                                                                 # 配送容量上限
    pi = 1000                                                                  # 一日あたりの配送単価
    h = 5                                                                      # 在庫単価
    b = 10                                                                     # 欠品単価
    W = 7                                                                      # １週間の日数
    
    """
    #補助変数の定義
    e = model.addVars(T, T, vtype=GRB.CONTINUOUS, lb=0, name="d") 
    d = model.addVars(T, T, vtype=GRB.CONTINUOUS, lb=0, name="d") 
    z = model.addVars(T, T, vtype=GRB.CONTINUOUS, lb=0, name="z") 
    v = model.addVars(T, T, vtype=GRB.CONTINUOUS, lb=0, name="v") 
    w = model.addVars(T, T, vtype=GRB.CONTINUOUS, lb=0, name="w") 
    """
        
    model.update() 
    model.setParam('TimeLimit', 60)  # 60秒でタイムリミット
    
    # 制約条件
    for t in range(T):
        model.addConstr(y[t] >= h * quicksum(q[s] - d[s]) for s in Upsilon ) 
        model.addConstr(I[t] <= M * delta[t])
        model.addConstr(o[t] <= M * (1 - delta[t]))
        i_t = df_input["day_index"].iloc[t] 
        model.addConstr(y[t] == s[i_t]) 
        model.addConstr(x[t] <= C * y[t])     # 配送容量制約
        
    # 目的関数
    model.setObjective(quicksum(pi * y[t] + h * I[t] + b * o[t] for t in range(T)), GRB.MINIMIZE) 
    model.optimize() 
    
    # 結果の出力
    x_values = [x[t].X for t in range(T)] 
    I_values = [I[t].X for t in range(T)] 
    o_values = [o[t].X for t in range(T)] 
    y_values = [y[t].X for t in range(T)] 
    delivery_costs = [pi * y[t].X for t in range(T)] 
    storage_costs = [h * I[t].X for t in range(T)] 
    shortage_costs = [b * o[t].X for t in range(T)] 
    
    for i in range(W):
        print(f"s*[{i}] = {s[i].X}") 
        
    delivery_schedule = [int(round(s[i].X)) for i in range(W)] return delivery_schedule    
    
    # 結果の出力
    x_values = [x[t].X for t in range(T)] 
    I_values = [I[t].X for t in range(T)] 
    o_values = [o[t].X for t in range(T)] 
    y_values = [y[t].X for t in range(T)] 
    delivery_costs = [pi * y[t].X for t in range(T)] 
    storage_costs = [h * I[t].X for t in range(T)] 
    shortage_costs = [b * o[t].X for t in range(T)] 
    
    # s_i をprintで出力
    for i in range(W):
        print(f"s*[{i}] = {delivery_schedule[i]}")
    
    # 結果をデータフレームとして格納
    df_results = pd.DataFrame({
        'Date': date_list,  # 日付のカラム
        'Demand': demand_list,
        'Order Quantity': x_values,
        'Inventory': I_values,
        'Shortage': o_values,
        'Delivery (y)': y_values,
        'Delivery Cost': delivery_costs,
        'Storage Cost': storage_costs,
        'Shortage Cost': shortage_costs,     
    })
    
    return df_results 

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
delivery_schedule = static_robust_optimization(training_data) 
# df_results = static_robust_optimization2(test_data, delivery_schedule) 
plot_order_quantity(df_results)  
export_results_to_csv(df_results) 