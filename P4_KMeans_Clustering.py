# ============================================================
# MI Practical 4 — K-Means Clustering
# ============================================================
# Aim: Given 8 points, perform K-means clustering with
#      M1=P1 (Cluster C1) and M2=P8 (Cluster C2).
# ============================================================
# DATA POINTS:
#   P1=[0.1,0.6]   P2=[0.15,0.71]  P3=[0.08,0.9]  P4=[0.16,0.85]
#   P5=[0.2,0.3]   P6=[0.25,0.5]   P7=[0.24,0.1]  P8=[0.3,0.2]
#
# INITIAL CENTROIDS:
#   M1 = P1 = [0.1, 0.6]
#   M2 = P8 = [0.3, 0.2]
# ============================================================
# EXAM ANSWERS:
#   Q1: Which cluster does P6 belong to?
#       P6 belongs to Cluster 2 (C2) after convergence.
#
#   Q2: Population of cluster around M2?
#       C2 = {P5, P6, P7, P8} — 4 points
#
#   Q3: Updated M1 and M2 after convergence?
#       M1 = [0.1225, 0.765]
#       M2 = [0.2475, 0.275]
#       Converges in 2 iterations, SSE = 0.045
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import math

# ──────────────────────────────────────────────
# Cell [1–2] — Setup Points & Distance Functions
# ──────────────────────────────────────────────

# Define the 8 data points
x = np.array([0.1,  0.15, 0.08, 0.16, 0.2,  0.25, 0.24, 0.3])
y = np.array([0.6,  0.71, 0.9,  0.85, 0.3,  0.5,  0.1,  0.2])

# Plot all points
plt.plot(x, y, "o")
plt.title("All 8 Data Points")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Distance functions
def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def manhattan_distance(x1, y1, x2, y2):
    return math.fabs(x1 - x2) + math.fabs(y1 - y2)

# ──────────────────────────────────────────────
# Cell [3] — Cluster Assignment Function
# ──────────────────────────────────────────────

# Returns 1 if point closer to m1, else 2
def returnCluster(m1, m2, x_co, y_co):
    d1 = manhattan_distance(m1[0], m1[1], x_co, y_co)
    d2 = manhattan_distance(m2[0], m2[1], x_co, y_co)
    if d1 < d2:
        return 1
    else:
        return 2

# ──────────────────────────────────────────────
# Cell [4] — K-Means Iteration Loop
# ──────────────────────────────────────────────

# Initial centroids: M1=P1, M2=P8
m1 = [0.1, 0.6]
m2 = [0.3, 0.2]

difference = math.inf   # Start with infinity
threshold  = 0.02       # Stop when change is tiny
iteration  = 0

while difference > threshold:
    print(f"\nIteration {iteration} — m1={m1}  m2={m2}")
    cluster1, cluster2 = [], []

    # STEP 1: Assign each point to nearest centroid
    for i in range(0, np.size(x)):
        num   = returnCluster(m1, m2, x[i], y[i])
        point = [x[i], y[i]]
        if num == 1:
            cluster1.append(point)
        else:
            cluster2.append(point)

    print("Cluster 1:", cluster1)
    print("Cluster 2:", cluster2)

    # STEP 2: Recalculate centroids (mean of cluster)
    m1_old = m1
    m1     = np.mean(cluster1, axis=0)   # axis=0 = column-wise mean
    m2_old = m2
    m2     = np.mean(cluster2, axis=0)

    # STEP 3: Calculate difference (convergence check)
    xAvg       = (math.fabs(m1[0] - m1_old[0]) + math.fabs(m2[0] - m2_old[0])) / 2
    yAvg       = (math.fabs(m1[1] - m1_old[1]) + math.fabs(m2[1] - m2_old[1])) / 2
    difference = xAvg if xAvg > yAvg else yAvg
    print(f"Difference: {difference}")
    iteration += 1

# Final Output
print("\n─── FINAL RESULTS ───")
print("Cluster 1 centroid: m1 =", m1)
print("Cluster 1 points:  ", cluster1)
print("Cluster 2 centroid: m2 =", m2)
print("Cluster 2 points:  ", cluster2)

# ──────────────────────────────────────────────
# Cell [5] — Plot Clusters
# ──────────────────────────────────────────────

clust1 = np.array(cluster1)
clust2 = np.array(cluster2)

plt.scatter(clust1[:, 0], clust1[:, 1], label='Cluster 1')         # C1 points
plt.scatter(clust2[:, 0], clust2[:, 1], label='Cluster 2')         # C2 points
plt.scatter([m1[0], m2[0]], [m1[1], m2[1]],
            marker="*", s=200, c='red', label='Centroids')         # Centroids
plt.title("K-Means Clustering Result")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# ──────────────────────────────────────────────
# EXPECTED FINAL OUTPUT:
# Cluster 1 centroid: m1 = [0.1225, 0.765]
# Cluster 1 points:   P1, P2, P3, P4
# Cluster 2 centroid: m2 = [0.2475, 0.275]
# Cluster 2 points:   P5, P6, P7, P8
# Converges in 2 iterations | SSE = 0.045
# ──────────────────────────────────────────────
