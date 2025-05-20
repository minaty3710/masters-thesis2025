# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 16:39:03 2025

@author: mina1
"""

import pandas as pd;
import matplotlib.pyplot as plt;
import os;

from  datetime import datetime;
from gurobipy import Model, GRB, quicksum


def adaptive_robust_optimization1(df_input):
    #インプットデータの読み込み
    demand_list = df_input['demand'].tolist();
    date_list = df_input['date'].tolist();
    
    #パラメータ設定
    T = len(demand_list);
    U = {t: list(range(t)) for t in range(T)}
    W = 7;
    h = 5;           # 保管コスト単価
    b = 10;          # 欠品コスト単価
    pi = 1000;       # 一日あたりの配送単価
    C = 500;         # 配送容量上限
    I0 = 0;          # 初期在庫
    Imax = 1500;     # 在庫容量上限
    M = 1e5;
    # Gurobi モデル
    model = Model();


    # 変数の定義
    x = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name= "x");           # 発注量
    I = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, ub=Imax, name="I");  # 期末在庫量
    y = model.addVars(T, vtype=GRB.BINARY, name="y");                     # 配送するか否か（バイナリ）
    o = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="o");           # 欠品量
    delta = model.addVars(T, vtype=GRB.BINARY, name="delta");
    z = model.addVars(T, T, vtype=GRB.CONTINUOUS, lb=0, name="z");        # アフィン係数
    v = model.addVars(T, T, vtype=GRB.CONTINUOUS, lb=0, name="v");           # 補助ベクトル
    s = model.addVars(W, vtype=GRB.BINARY, name="s");                     # 各曜日に配送するか否か（バイナリ）
    
    model.update();
    model.setParam('TimeLimit', 60)  # 例えば60秒でタイムリミット
    
    # 制約条件
    for t in range(T):
        if t == 0:
            model.addConstr(I[t] - o[t] == z[t,0]  + quicksum(v[t, u] * demand_list[u] for u in U[t]) - demand_list[t]);            #　在庫の更新(t=0)
        else:
            model.addConstr(I[t] - o[t] == I[t-1] + z[t,0] + quicksum(v[t, u] * demand_list[u] for u in U[t]) - demand_list[t]);  #　在庫の更新(t≠0)
            
        model.addConstr(x[t] == z[t,0] + quicksum(z[t,u]*demand_list[u] for u in range(t)));      # アフィン関数の定義（t≠0)
        model.addConstr(I[t] <= M * delta[t])
        model.addConstr(o[t] <= M * (1 - delta[t]))
        i_t = df_input["day_index"].iloc[t];
        model.addConstr(y[t] == s[i_t]);
        model.addConstr(z[t,0] + quicksum(v[t, u] * demand_list[u] for u in U[t]) <= C * y[t]);    # 配送容量制約
        
        
        #補助ベクトルの定義
        for t in range(T):
            for u in range(T):
                if u in U[t]:
                    model.addConstr(v[t, u] == z[t, u])
                else:
                    model.addConstr(v[t, u] == 0)
        
    # 目的関数
    model.setObjective(quicksum(pi * y[t] + h * I[t] + b * o[t] for t in range(T)), GRB.MINIMIZE);
    model.optimize();
    
    if model.Status == GRB.INFEASIBLE:
        print("モデルは実行不能です。IIS を出力します。")
        model.computeIIS()
        model.write("model.ilp")  # IISの内容をILPファイルに書き出す

    
    for i in range(W):
        print(f"s*[{i}] = {s[i].X}");
        
    delivery_schedule = [int(round(s[i].X)) for i in range(W)];
    
    return delivery_schedule;   

def adaptive_robust_optimization2(df_input, delivery_schedule):
    #インプットデータの読み込み
    demand_list = df_input['demand'].tolist();
    date_list = df_input['date'].tolist();
    
    #パラメータ設定
    T = len(demand_list);
    U = {t: list(range(t)) for t in range(T)}
    W = 7;
    h = 5;       # 保管コスト単価
    b = 10;       # 欠品コスト単価
    pi = 1000;      # 一日あたりの配送単価
    C = 500;      # 配送容量上限
    I0 = 0;        # 初期在庫
    Imax = 1500;   # 在庫容量上限
    M = 1e5;
    
    # Gurobi モデル
    model = Model();
    
    # 変数の定義
    x = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name= "x");           # 発注量
    I = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, ub=Imax, name="I");  # 期末在庫量
    y = model.addVars(T, vtype=GRB.BINARY, name="y");                     # 配送するか否か（バイナリ）
    o = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="o");           # 欠品量
    delta = model.addVars(T, vtype=GRB.BINARY, name="delta");
    z = model.addVars(T, T, vtype=GRB.CONTINUOUS, lb=0, name="z");        # アフィン係数
    v = model.addVars(T, T, vtype=GRB.CONTINUOUS, lb=0, name="v");           # 補助ベクトル
     
    model.update();
    model.setParam('TimeLimit', 60)  # 例えば60秒でタイムリミット
    
    # 制約条件
    for t in range(T):
        if t == 0:
            model.addConstr(I[t] - o[t] == z[t,0] + quicksum(v[t, u] * demand_list[u] for u in U[t]) - demand_list[t]);            #　在庫の更新(t=0)
        else:
            model.addConstr(I[t] - o[t] == I[t-1] + z[t,0] + quicksum(v[t, u] * demand_list[u] for u in U[t]) - demand_list[t]);  #　在庫の更新(t≠0)
            
        model.addConstr(x[t] == z[t,0] + quicksum(z[t,u] * demand_list[u] for u in range(t)));      # アフィン関数の定義（t≠0)
        model.addConstr(I[t] <= M * delta[t])
        model.addConstr(o[t] <= M * (1 - delta[t]))
        i_t = df_input["day_index"].iloc[t];
        model.addConstr(y[t] <= delivery_schedule[i_t])
        model.addConstr(z[t,0] + quicksum(v[t, u] * demand_list[u] for u in U[t]) <= C * y[t]);    # 配送容量制約
        
        
        #補助ベクトルの定義
        for t in range(T):
            for u in range(T):
                if u in U[t]:
                    model.addConstr(v[t, u] == z[t, u])
                else:
                    model.addConstr(v[t, u] == 0)
        
    # 目的関数
    model.setObjective(quicksum(pi * y[t] + h * I[t] + b * o[t] for t in range(T)), GRB.MINIMIZE);
    model.optimize();
    
    # 結果の出力
    x_values = [x[t].X for t in range(T)];
    I_values = [I[t].X for t in range(T)];
    o_values = [o[t].X for t in range(T)];
    y_values = [y[t].X for t in range(T)];
    v_values = [[v[t, u].X for u in range(T)] for t in range(T)]  # 2次元変数の出力
    z_values = [[z[t, u].X for u in range(T)] for t in range(T)]  # 2次元変数の出力
    delivery_costs = [pi * y[t].X for t in range(T)];
    storage_costs = [h * I[t].X for t in range(T)];
    shortage_costs = [b * o[t].X for t in range(T)];
    
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
        'v_values': [str(v_values[t]) for t in range(T)],  # 2次元配列を文字列に変換して出力
        'z_values': [str(z_values[t]) for t in range(T)],  # 2次元配列を文字列に変換して出力
        'Delivery Cost': delivery_costs,
        'Storage Cost': storage_costs,
        'Shortage Cost': shortage_costs,
    })
    
    return df_results;

def plot_order_quantity(df_results):
    #リザルトデータの読み込み
    date_list = df_results['Date'].tolist();
    x_values = df_results['Order Quantity'].tolist();
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


read_path = "C:\\Users\mina1\.spyder-py3\master's thesis\dataset\demand_data_2025-04-22_1416.csv";
df_input = pd.read_csv(read_path);
# 日付が datetime 型でなければ変換
df_input["date"] = pd.to_datetime(df_input["date"]);
# 月曜日を0とした曜日インデックスを列に追加
df_input["day_index"] = df_input["date"].dt.weekday;
# training_data/test_data にもこの列を継承
training_data = df_input[(df_input["date"].dt.year == 2025) & (df_input["date"].dt.month <= 2)].copy();
test_data = df_input[(df_input["date"].dt.year == 2025) & (df_input["date"].dt.month >= 3)].copy();
delivery_schedule = adaptive_robust_optimization1(training_data);
df_results = adaptive_robust_optimization2(test_data, delivery_schedule);
plot_order_quantity(df_results); 
export_results_to_csv(df_results);