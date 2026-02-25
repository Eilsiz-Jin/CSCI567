import numpy as np
import matplotlib.pyplot as plt

# --- Data and model from previous parts ---
# original training points (x, y)
xs = np.array([1.0, -1.0, 0.0])
ys = np.array([-1.0, -1.0, 1.0])

# feature map phi(x) = [x, x^2]^T
Phi = np.vstack([xs, xs**2]).T  # shape (3,2)

# optimal primal solution from 1.5
w_star = np.array([0.0, -2.0]).reshape(2, 1)   # column vector (2,1)
b_star = 1.0

# support vector indices (from alphas: alpha=[1,1,2] -> all positive)
sv_idx = [0, 1, 2]

# --- Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(12,5))

# Left: feature space (2D) with hyperplane and margins
ax = axes[0]
ax.set_title("Feature space (phi(x) = [x, x^2])")
ax.set_xlabel("z1 = x")
ax.set_ylabel("z2 = x^2")

# plot training points in feature space, colored by label
for i in range(len(xs)):
    z = Phi[i]
    label = ys[i]
    color = 'tab:blue' if label > 0 else 'tab:red'
    ax.scatter(z[0], z[1], c=color, s=70, edgecolor='k')
    ax.text(z[0]+0.03, z[1]+0.03, f"({xs[i]:.0f}, y={int(label)})", fontsize=9)

# draw decision line: w^T z + b = 0  -> z2 as function of z1 if possible
# Solve w1 * z1 + w2 * z2 + b = 0  => z2 = -(w1*z1 + b)/w2  (if w2 != 0)
w1, w2 = w_star.flatten()
if abs(w2) > 1e-12:
    z1_vals = np.linspace(-1.5, 1.5, 400)
    z2_vals = -(w1 * z1_vals + b_star) / w2
    ax.plot(z1_vals, z2_vals, 'k-', label='decision plane (w^T z + b = 0)')
    # margins: w^T z + b = ±1
    z2_margin_pos = -(w1 * z1_vals + b_star - 1.0) / w2
    z2_margin_neg = -(w1 * z1_vals + b_star + 1.0) / w2
    ax.plot(z1_vals, z2_margin_pos, 'k--', linewidth=1, label='margin ±1')
    ax.plot(z1_vals, z2_margin_neg, 'k--', linewidth=1)
else:
    # If w2 == 0 we'd param by z2; not the case here.
    pass

# highlight support vectors
for i in sv_idx:
    z = Phi[i]
    ax.scatter(z[0], z[1], facecolors='none', edgecolors='black', s=250, linewidths=2)

ax.legend(loc='upper right')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-0.5, 2.5)
ax.grid(True)

# Right: original 1D space
ax2 = axes[1]
ax2.set_title("Original input space")
ax2.set_xlabel("x")
ax2.set_ylabel("f(x) = w^T phi(x) + b")

x_plot = np.linspace(-1.5, 1.5, 400)
phi_x = np.vstack([x_plot, x_plot**2]).T  # (400,2)
f_vals = (phi_x.dot(w_star)).flatten() + b_star  # shape (400,)

ax2.plot(x_plot, f_vals, label='f(x)=w^T phi(x)+b')
ax2.axhline(0, color='k', linewidth=0.8)
# mark decision boundaries x = ±1/sqrt(2)
x_bd = 1.0 / np.sqrt(2.0)
ax2.axvline(x_bd, color='gray', linestyle='--')
ax2.axvline(-x_bd, color='gray', linestyle='--')
# plot original training points on x-axis with their labels
for i in range(len(xs)):
    ax2.scatter(xs[i], 0.0, c='tab:blue' if ys[i] > 0 else 'tab:red', s=70, edgecolor='k')
    ax2.text(xs[i]+0.03, 0.06, f"y={int(ys[i])}", fontsize=9)

ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-2.5, 2.0)
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
