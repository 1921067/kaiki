import pandas as pd

df = pd.read_csv('all_players.csv')

df.columns = ['Player', 'Club', 'POS', 'GP', 
              'GS', 'MINS', 'G', 'A', 'SHTS', 
              'SOG', 'GWG', 'PKG/A', 'HmG', 'RdG',
              'G/90min', 'SC%', 'GWA', 'HmA', 'RdA', 'A/90min',
              'FC', 'FS', 'OFF', 'YC', 'RC', 'SOG%',
              'Year', 'Season' ]
df.head()

import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix

cols = ['MINS', 'G', 'A', 'SHTS', 'SOG']

scatterplotmatrix(df[cols].values, figsize=(10, 8), 
                  names=cols, alpha=0.8)
plt.tight_layout()
#plt.savefig('images/10_03.png', dpi=300)
plt.show()


import numpy as np
from mlxtend.plotting import heatmap


cm = np.corrcoef(df[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)

# plt.savefig('images/10_04.png', dpi=300)
plt.show()




X = df[['SHTS']].values
y = df['G'].values


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

ransac = RANSACRegressor(LinearRegression(), 
                         max_trials=100, 
                         min_samples=50, 
                         loss='absolute_loss', 
                         residual_threshold=5.0, 
                         random_state=0)


ransac.fit(X, y)



inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(0, 180, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='steelblue', edgecolor='white', 
            marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='limegreen', edgecolor='white', 
            marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='black', lw=2)   
plt.xlabel('Shoots [SHTS]')
plt.ylabel('Goal [G]')
plt.legend(loc='upper left')


plt.show()

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
ransac.fit(X_std, y_std)



print("シュート数を入力してください。")
sh = input()
num_shoot_std = sc_x.transform(np.array([[sh]]))
goal_std = ransac.predict(num_shoot_std)
print("%.0fゴール" % np.round(sc_y.inverse_transform(goal_std)))
