import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

def solve_minimum_volume_ellipsoid(points):
    m, n = points.shape
    model = gp.Model("minimum_volume_ellipsoid")
    model.setParam("OutputFlag", 0)  # ログ出力をOFF

    # 変数 A ∈ ℝ^{n×n}, b ∈ ℝ^n
    A = model.addVars(n, n, name="A", lb=-GRB.INFINITY)
    b = model.addVars(n, name="b", lb=-GRB.INFINITY)

    # Aを対称行列にする制約
    for i in range(n):
        for j in range(i+1, n):
            model.addConstr(A[i, j] == A[j, i])

    # 各点が楕円体内にあるように制約を課す: ||A x_i + b||² ≤ 1
    for idx in range(m):
        quad_expr = gp.QuadExpr()
        for i in range(n):
            affine = gp.LinExpr()
            for j in range(n):
                affine += A[i, j] * points[idx, j]
            affine += b[i]
            quad_expr += affine * affine
        model.addQConstr(quad_expr <= 1)

    # 目的関数（Aのトレース最小化 ≒ 体積の対数最小化の近似）
    model.setObjective(gp.quicksum(A[i, i] for i in range(n)), GRB.MINIMIZE)

    model.optimize()

    # 結果取得
    A_val = np.array([[A[i, j].X for j in range(n)] for i in range(n)])
    b_val = np.array([b[i].X for i in range(n)])

    # 2次元可視化
    if n == 2:
        fig, ax = plt.subplots()
        ax.scatter(points[:, 0], points[:, 1], c='black', label='Points')

        # 楕円体の描画（単位円を変換）
        theta = np.linspace(0, 2*np.pi, 100)
        circle = np.vstack([np.cos(theta), np.sin(theta)])
        ellipse = np.linalg.inv(A_val) @ (circle - b_val.reshape(-1, 1))
        ax.plot(ellipse[0, :], ellipse[1, :], color='#2F5597', label='Ellipsoid')

        ax.set_aspect('equal')
        ax.legend()
        plt.title("Minimum Volume Ellipsoid")
        plt.grid(True)
        plt.show()

    return A_val, b_val

# 実行例（テスト）
if __name__ == "__main__":
    np.random.seed(0)
    pts = np.random.randn(30, 2) * 0.5 + np.array([2, 1])  # 2次元の点を30個生成
    A, b = solve_minimum_volume_ellipsoid(pts)
