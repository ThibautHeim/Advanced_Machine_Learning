import numpy as np
import matplotlib.pyplot as plt
import torch
from flow import Flow


def color_mapping(Latent_variables, Labels, requirement = None):
    colors = ["#1f77b4","#ff7700","#009a00","#ff0303","#690fbe","#814236","#e050b5","#0000ff","#efe30f","#14d1e6"]
    fig, axs = plt.subplots(figsize=(8, 6))
    for label in range(10):
        mask = Labels == label
        masked_Latent = Latent_variables[mask]
        if requirement == None :
            axs.scatter(masked_Latent[:,0], masked_Latent[:,1], alpha=0.7, s=6, c=colors[label], label=label)
        else : 
            if label in requirement :
                axs.scatter(masked_Latent[:,0], masked_Latent[:,1], alpha=0.7, s=6, c=colors[label], label=label)
    axs.legend(loc='upper right')
    axs.set_title("Latent variables projected on the 2 principal components for different labels", fontweight='bold', fontsize = 10)
    plt.show()


def grid_for_distribution_plot(dist, latent_dim, min = -4, max = 4):
    assert latent_dim == 2, "Impossible to plot the prior which is too high dimensional"
    # Create a 2D grid
    x = torch.linspace(min, max, 200)
    y = torch.linspace(min, max, 200)
    X, Y = torch.meshgrid(x, y, indexing='xy')

    # Stack the coordinates for the density
    grid = torch.stack([X, Y], dim=-1)  # shape: (200, 200, 2)
    try : 
        if isinstance(dist, Flow) :
            flat_grid = grid.reshape(-1, 2)           

            with torch.no_grad():
                log_prob = dist.log_prob(flat_grid)   # (40000, 2)
                log_prob = log_prob.reshape(200, 200)  # (200, 200)
                
        else : 
            log_prob = dist.log_prob(grid)
    except(ValueError):
        print("Latent-space dimension is not suited for this problem")
    pdf = torch.exp(log_prob)
    
    return pdf



if __name__ == "__main__" : 
    dummy_latent = np.random.rand(100, 2)
    dummy_labels = np.random.randint(0, 10, size=(100,))

    color_mapping(dummy_latent,dummy_labels, requirement=[1,2,7])

