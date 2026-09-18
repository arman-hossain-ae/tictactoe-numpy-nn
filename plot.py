import numpy as np
import matplotlib.pyplot as plt

log = np.load("training_log.npz", allow_pickle=True)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(log["step"], log["avg_loss"], color="#E53935", linewidth=2, label="avg_loss")
ax.set_title("Training Loss Over Time", fontsize=14, weight="bold")
ax.set_xlabel("Steps")
ax.set_ylabel("avg_loss")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend()
fig.tight_layout()
fig.savefig("assets/training_curve.png", dpi=150)
print("Saved assets/training_curve.png")
