import pandas as pd
import numpy as np
from datetime import datetime

start_date = "2024-01-01"
end_date = "2025-01-31"
dates = pd.date_range(start = start_date, end = end_date, freq = "D")

mu = {
      0: 150,
      1: 140,
      2: 130,
      3: 140,
      4: 150,
      5: 200,
      6: 250,
     }

sigma = {
    0: 15,
    1: 15,
    2: 15,
    3: 15,
    4: 15,
    5: 20,
    6: 25
    }

#需要生成
data = []
for t in dates:
    i = t.weekday()
    demand = max(0, np.random.normal(mu[i], sigma[i]))
    data.append((t.strftime('%Y-%m-%d'), int(demand)))
#データフレームにして保存
df = pd.DataFrame(data, columns = ['date', 'demand'])
now = datetime.now()
timestamp = now.strftime('%Y-%m-%d_%H%M')
filename = f'demand_data_{timestamp}.csv'
save_path = f"C:\\Users\mina1\.spyder-py3\master's thesis\dataset\{filename}"
df.to_csv(save_path, index = False)
print(save_path)

