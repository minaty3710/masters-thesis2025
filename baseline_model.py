import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from  datetime import datetime
from gurobipy import Model, GRB, quicksum


def baseline_model(df_input):   
    # Gurobi モデル
    model = Model()
    
    #パラメータ設定
    D = df_input['demand'].tolist()                                            # 期𝑡の需要量 
    T = len(D)                                                                 # 全期間
    Upsilon = {t: list(range(t)) for t in range(T)}                            # 𝑡−1期までの集合
    d_mean = np.mean(D)                                                        # 期𝑡の需要量の平均
    Imax = 1500                                                                # 店舗の在庫上限
    Qmax = 500                                                                 # 配送容量上限
    pi = 1000                                                                  # 一日あたりの配送単価
    h = 5                                                                      # 在庫単価
    b = 10                                                                     # 欠品単価
    W = 7                                                                      # １週間の日数
    
    # 決定変数の定義
    q = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="q")                 # 期𝑡の発注量
    y = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="y")                 # 期𝑡の在庫コスト（欠品コスト）  
    delta = model.addVars(T, vtype=GRB.BINARY, name="delta")                   # 配送有無（1のとき配送を実施）
    sigma = model.addVars(T, vtype=GRB.BINARY, name="sigma")                   # 各曜日の配送有無（1の曜日は配送可能）
    
    """
    #補助変数の定義
    e = model.addVars(T, T, lb=0, vtype=GRB.CONTINUOUS, lb=0, name="e") 
    d = model.addVars(T, T, lb=0, vtype=GRB.CONTINUOUS, lb=0, name="d") 
    z = model.addVars(T, T, lb=0, vtype=GRB.CONTINUOUS, lb=0, name="z") 
    v = model.addVars(T, T, lb=0, vtype=GRB.CONTINUOUS, lb=0, name="v") 
    w = model.addVars(T, T, lb=0, vtype=GRB.CONTINUOUS, lb=0, name="w") 
    """
    
    model.update() 
    model.setParam('TimeLimit', 60)  # 60秒でタイムリミット
    
    # 制約条件
    for t in range(T):
        model.addConstr(y[t] >= h * quicksum(q[s] - D[s] for s in range(t+1)))
        model.addConstr(y[t] >= b * quicksum(D[s] - q[s] for s in range(t+1)))
        model.addConstr(quicksum(q[s] - D[s] for s in range(t+1)) <= Imax)
        model.addConstr(q[t] <= Qmax * delta[t])    
        i_t = df_input["day_index"].iloc[t] 
        model.addConstr(delta[t] == sigma[i_t]) 

    # 目的関数
    model.setObjective(quicksum(y[t] + pi * delta[t] for t in range(T)), GRB.MINIMIZE) 
    model.optimize() 
    
    # 結果の出力
    q_values = [q[t].X for t in range(T)] 
    y_values = [y[t].X for t in range(T)] 
    delta_values = [delta[t].X for t in range(T)] 
    sigma_values = [sigma[df_input["day_index"].iloc[t]].X for t in range(T)] 
    inventory = [max(0, sum(q[s].X - D[s] for s in range(t+1))) for t in range(T)]
    out_of_stock = [max(0, sum(D[s] - q[s].X for s in range(t+1))) for t in range(T)]
    delivery_costs = [pi * delta[t].X for t in range(T)] 
    inventory_costs = [h * inventory[t] for t in range(T)] 
    out_of_stock_costs = [b * out_of_stock[t] for t in range(T)] 
    
    """"
    for i in range(W):
        print(f"sigma*[{i}] = {sigma[i].X}") 
    """
    # 結果をデータフレームとして格納
    date_list = df_input['date'].tolist()
    weekday_list = df_input['date'].dt.strftime('%a').tolist()       
    df_results = pd.DataFrame({
        'Date': date_list,  
        'week_day': weekday_list,
        'Demand': D,
        'Order Quantity': q_values,
        'y_Cost' : y_values,
        'Inventory': inventory,
        'Shortage': out_of_stock,
        'delta': delta_values,
        'sigma' : sigma_values,
        'Delivery Cost': delivery_costs,
        'Inventory Cost': inventory_costs,
        'Shortage Cost': out_of_stock_costs,
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
    filename = f'baseline_model_result_{timestamp}.png'
    save_path = os.path.join("C:/Users/mina1/.spyder-py3/master's thesis/result", filename)
    plt.savefig(save_path, dpi=300)
    plt.show()
    
def export_results_to_csv(df_results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f'baseline_model_result_{timestamp}.csv'
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
#training_data = df_input[(df_input["date"].dt.year == 2025) & (df_input["date"].dt.month <= 2)].copy() 
test_data = df_input[(df_input["date"].dt.year == 2025) & (df_input["date"].dt.month >= 1)].copy() 
df_results = baseline_model(test_data)  
plot_order_quantity(df_results)
export_results_to_csv(df_results)