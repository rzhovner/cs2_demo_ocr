import sys
import pandas as pd
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else "training1_engagements.csv"
window = int(sys.argv[2]) if len(sys.argv) > 2 else 10

df = pd.read_csv(csv_path)
df["rolling_hit_rate"] = df["hit_rate"].rolling(window, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(14, 5))

ax.bar(df["engagement_id"], df["hit_rate"], color="#aed6f1", alpha=0.6, label="Per-engagement hit rate")

ax.plot(df["engagement_id"], df["rolling_hit_rate"], color="#1a5276", linewidth=2,
        label=f"{window}-engagement rolling avg")

overall = df["hit_rate"].mean()
ax.axhline(overall, color="#e74c3c", linestyle="--", linewidth=1, label=f"Session avg ({overall:.2f})")

ax.set_xlabel("Engagement #")
ax.set_ylabel("Hit Rate")
ax.set_title(f"Hit Rate — Rolling Average (window={window})")
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", alpha=0.3)

out_path = csv_path.replace(".csv", "_hitrate.png")
plt.tight_layout()
plt.savefig(out_path, dpi=150)
plt.show()
print(f"Saved: {out_path}")
