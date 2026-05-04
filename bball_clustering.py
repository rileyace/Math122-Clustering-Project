import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv("all_batters.csv")

# Constructed feature to directly encode a preference for hitting against lefties or righties
df['woba_diff'] = df['wOBA_left'] - df['wOBA_right']

# Another constructed feature
df['ops_diff'] = df['OPS_left'] - df['OPS_right']

# X = df[['wOBA_left', 'wOBA_right']].dropna()
# X = df[['wOBA_left', 'wOBA_right', 'woba_diff']].dropna()
X = df[['ops_diff', 'woba_diff']].dropna()

kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(X)

#----------- plotting with only woba_left vs. woba_right -------------
#
# plt.figure()
# plt.scatter(X['wOBA_left'], X['wOBA_right'], c=clusters)
#
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1], marker='x')
#
# plt.xlabel("wOBA vs Left-Handed Pitchers")
# plt.ylabel("wOBA vs Right-Handed Pitchers")
# plt.title("KMeans Clusters of Hitters")
#
# plt.show()
# ----------------------------------------------------------------

# ---------------- plotting using split_diff ----------------------
# df_clean = df.loc[X.index].copy()
# df_clean['cluster'] = clusters
#
# # Plot only woba_left vs woba_right, colored by cluster
# plt.figure()
# plt.scatter(df_clean['wOBA_left'], df_clean['wOBA_right'], c=df_clean['cluster'])
#
# centroids = kmeans.cluster_centers_
#
# # Centroids are now 3D, but plot only first two dimensions
# plt.scatter(centroids[:, 0], centroids[:, 1], marker='x')
#
# plt.xlabel("wOBA vs Left-Handed Pitchers")
# plt.ylabel("wOBA vs Right-Handed Pitchers")
# plt.title("KMeans Clusters Using wOBA Left, wOBA Right, and wOBA Difference")
# plt.show()
# ------------------------------------------------------------------

#------------------ plotting using ops_diff vs. woba_diff -----------
plt.figure()
plt.scatter(X['ops_diff'], X['woba_diff'], c=clusters)

centroids = kmeans.cluster_centers_

plt.scatter(centroids[:, 0], centroids[:, 1], marker='x')

plt.xlabel("OPS_diff")
plt.ylabel("wOBA_diff")
plt.title("KMeans Clusters Using OPS difference and wOBA difference")
plt.show()
#-----------------------------------------------------------------------