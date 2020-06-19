# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)
N = 20
X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
                    np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]
X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
bins = np.linspace(-5, 10, 10)
print(X)

plt.hist(X[:, 0], bins=bins, fc='#AAAAFF', density=True)
print(X[:, 0])

kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)
log_dens = kde.score_samples(X_plot)

print(X_plot[:, 0])
plt.fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')


