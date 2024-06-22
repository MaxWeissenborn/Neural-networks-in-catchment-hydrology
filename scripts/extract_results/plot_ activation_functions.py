import numpy as np
from settings_plot import *
import os

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def linear(x):
    return x

def leakyRelu(x):
    return np.maximum(0.3 * x, x)


prefix = 2
# Generate x values
x = np.linspace(-10, 10, 100)

# Create subplots
fig, axs = plt.subplots(ncols=3, figsize=(18, 6), sharex="all")

# Linear plot
a = sns.lineplot(x=x, y=linear(x), ax=axs[0], color="#176D9C")
a.set_title('a) Linear Activation Function', weight="bold")
a.set_yticks([-10, -5,  0,  5, 10])


# Relu plot
b = sns.lineplot(x=x-0.15, y=relu(x), ax=axs[1], color="#176D9C",  label='ReLU')
axs[1].set_title('b) ReLU Activation Functions', weight="bold")

# LeakyRelu plot
sns.lineplot(x=x, y=leakyRelu(x), ax=axs[1], color="#2EABB8", label='LeakyReLU')
b.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95), )

# Sigmoid plot
c = sns.lineplot(x=x, y=sigmoid(x), ax=axs[2], color="#176D9C")
axs[2].set_title('c) Sigmoid Activation Function', weight="bold")

for ax in axs:
    ax.set_ylabel("y", labelpad=20, weight="bold", rotation=360)
    ax.set_xlabel("x", labelpad=20, weight="bold")

fig.tight_layout(pad=1.0)
plt.subplots_adjust(wspace=0.18)
sns.despine(right=True, left=True, bottom=True)

# Display the plot
fig.savefig(os.path.join("../../results/Activation_Functions-%s.pdf" % prefix), format="pdf",
            bbox_inches="tight")
plt.close(fig)
plt.show()
