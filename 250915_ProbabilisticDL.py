import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

import tensorflow as tf
import tensorflow_probability as tfp

print(tf.__version__)

# # Unfair die: faces 1-5 have p, face 6 has 2p => 5p + 2p = 1 => p = 1/7
# probs = [1/7]*5 + [2/7]
#
# # 1. Define categorical distribution for single rolls
# die_dist = tfp.distributions.Categorical(probs=probs)
#
# n_trials = 100
# rolls_in_trial = 10
#
# # 2. Generate a sequence of 10 rolls (values 0-5; add 1 for face labels)
# ten_rolls = die_dist.sample(10).numpy() + 1
# print("Sequence of 10 rolls:", ten_rolls)
#
# # 3. Generate 100 rolls and plot empirical frequencies
# multi = tfp.distributions.Multinomial(total_count=rolls_in_trial, probs=probs)
# counts_per_trial = multi.sample(n_trials).numpy()       # shape (100,6)
# print("Counts per trial shape:", counts_per_trial.shape)
# print("First trial counts (faces 1..6):", counts_per_trial[0])
#
# # Plot distribution of number of sixes across trials
# num_sixes = counts_per_trial[:, -1]
# plt.figure(figsize=(6,4))
# sns.countplot(x=num_sixes)
# plt.xlabel("Number of sixes in a 10-roll trial")
# plt.ylabel("Frequency (over 100 trials)")
# plt.title("Distribution of six counts across trials")
# plt.show()


# Plotting binomial distribution for increasing number of trials
n_list = [10, 50, 100, 500]

plt.figure(figsize=(10,6))
for n in n_list:
    k = tf.range(0, n+1, dtype=tf.float32)
    dist = tfp.distributions.Binomial(total_count=n, probs=0.5)
    pmf = dist.prob(k)
    plt.plot(k.numpy(), pmf.numpy(), marker='o', linewidth=1, markersize=3, label=f'n={n}')

plt.title('Binomial pmf: count of success (p=0.5)')
plt.xlabel('k (number of successes)')
plt.ylabel('P(K = k)')
plt.legend()
plt.tight_layout()
plt.show()