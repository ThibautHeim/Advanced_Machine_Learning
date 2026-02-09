import numpy as np
import matplotlib.pyplot as plt

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
    plt.show()


if __name__ == "__main__" : 
    dummy_latent = np.random.rand(100, 2)
    dummy_labels = np.random.randint(0, 10, size=(100,))

    color_mapping(dummy_latent,dummy_labels, requirement=[1,2,7])

