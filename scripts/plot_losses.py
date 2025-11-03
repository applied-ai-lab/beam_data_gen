import numpy as np
import copy

import numpy as np
from matplotlib import pyplot as plt 

import scienceplots

plt.style.use(['science', 'ieee','no-latex'])


def main():
    
    # Load data
    path = "data/beam_losses/beam_loss.npy"
    
    losses = np.load(path)[:, 1:]
    
    no_runs, no_steps = losses.shape
    
    loss_min = np.zeros(no_steps)
    loss_max = np.zeros(no_steps)
    
    for k in range(no_steps):
        loss_min[k] = np.min(losses[:, k])
        loss_max[k] = np.max(losses[:, k])        
    
    timesteps = np.arange(0, no_steps)
    
    max_value = copy.deepcopy(loss_max[0])
    
    # Normalise losses
    losses /= max_value
    loss_max /= max_value
    loss_min /= max_value
    
    colour = "#0066ff"
    fill_colour = "#b3b3ff"
    
    fig, ax = plt.subplots()
    fig.set_size_inches(3 * 1.618, 3)
    fig.set_dpi(300)
    
    plt.plot(timesteps, losses.transpose(), color=colour, alpha=0.7)
    plt.fill_between(timesteps, loss_min, loss_max,
                 color=fill_colour, alpha=0.6)
    
    ax.set_xlabel("Gradient Steps")
    ax.set_ylabel("Goal Loss")
    
    plt.grid()
    plt.show()
    
    
    return 0


if __name__ == "__main__":
    main()
