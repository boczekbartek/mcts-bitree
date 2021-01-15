# %%
import pandas as pd
import matplotlib.pyplot as plt
import os

plots_dir = "mcts_plots"
os.makedirs(plots_dir, exist_ok=True)
log1 = "test_c_param1.sh.log"
log = "test_c_param.sh.log"
df1 = pd.read_csv(log, sep="\t", header=None)
dfsm = pd.read_csv(log1, sep="\t", header=None)
df = pd.concat([dfsm, df1])
df.columns = ["c", "win", "gap"]
grp = df.groupby(by="c").mean()

grp.plot()
plt.title("Average diff between a MCTS-found leaf and the best leaf")
plt.ylabel("best_leaf - mcts_leaf")
min_i = grp["gap"].argmin()
min_c = grp.index[min_i]
min_gap = grp.gap[min_c]
plt.scatter(min_c, min_gap, s=30)
plt.text(min_c, min_gap + 1, s=f"C={min_c}")
plt.savefig(f"{plots_dir}/call")

df = df1
df.columns = ["c", "win", "gap"]
grp = df.groupby(by="c").mean()
plt.figure()
df.plot()
plt.title("C = 1:40 | Average diff between a MCTS-found leaf and the best leaf")
plt.ylabel("best_leaf - mcts_leaf")
min_i = grp["gap"].argmin()
min_c = grp.index[min_i]
min_gap = grp.gap[min_c]
plt.scatter(min_c, min_gap, s=30)
plt.text(min_c + 1, min_gap, s=f"C={min_c}")
plt.savefig(f"{plots_dir}/csbig")

df = dfsm
df.columns = ["c", "win", "gap"]
grp = df.groupby(by="c").mean()
plt.figure()
df.plot()
plt.title("C = 0.1:2 | Average diff between a MCTS-found leaf and the best leaf")
plt.ylabel("best_leaf - mcts_leaf")
min_i = grp["gap"].argmin()
min_c = grp.index[min_i]
min_gap = grp.gap[min_c]
plt.scatter(min_c, min_gap, s=30)
plt.text(min_c, min_gap + 1, s=f"C={min_c}")
plt.savefig(f"{plots_dir}/csmall")
# %%


# %%
