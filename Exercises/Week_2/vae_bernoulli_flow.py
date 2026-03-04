# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm

# Other imported modules (exercises)
from sklearn.decomposition import PCA
import numpy as np
from math import floor
import json
import os

# Flow related classes
from flow import MaskedCouplingLayer, Flow







class GaussianBase(nn.Module): # p(z) = N(0, I)
    def __init__(self, M):
        """
        Define a Gaussian distribution with zero mean and unit variance. 
        It can be used as prior base distribution for flow based prior

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianBase, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)
    
    def log_prob(self, x):
        return self.forward().log_prob(x)
    

    

class GaussianMixturePrior(nn.Module): # p(z) = sum_i w_i N(z|locs[i], scales[i])
    def __init__(self, M, categorical_weights = torch.ones(10,), locs = None, scales = None):
        """
        Define a Gaussian Mixture distribution.

        Parameters:
        categorical_weights: [torch.Tensor] 
           A tensor of dimension `(num_components,)` representing the weights of the mixture components.
        means: [torch.Tensor] 
           A tensor of dimension `(num_components, M)` representing the means of the Gaussian components, where M is the dimension of the latent space.
        stds: [torch.Tensor] 
           A tensor of dimension `(num_components, M)` representing the standard deviations of the Gaussian components.
        """
        super(GaussianMixturePrior, self).__init__()
        # Input arguments
        self.M = M
        self.categorical_weights = nn.Parameter(categorical_weights)
        if locs is not None :
            self.locs = nn.Parameter(locs) # tensor of nb_gaussians loc vectors
        else :
            self.locs = nn.Parameter(torch.zeros(10,self.M))
        if scales is not None :    
            self.scales = nn.Parameter(scales) # tensor of nb_gaussians scale vectors
        else :
            self.scales = nn.Parameter(torch.ones(10,self.M))
        self.nb_gaussians = len(self.locs)
    
    def get_distribution(self):
        gaussians = td.Independent(td.Normal(self.locs, self.scales), 1)
        categorical = td.Categorical(logits=self.categorical_weights) 
        return td.MixtureSameFamily(mixture_distribution=categorical, component_distribution=gaussians)
    
    def log_prob(self, x):
        return self.get_distribution().log_prob(x)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return self.get_distribution()

    



class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)
    



class BernoulliDecoder(nn.Module): # p(x|z) = Bernoulli(logits=decoder_net(z))
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)
    

    
class MultivariateGaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        
        super(MultivariateGaussianDecoder, self).__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        means = torch.flatten(self.decoder_net(z)[:,0,:], start_dim=1)
        variances = torch.flatten(self.decoder_net(z)[:,1,:], start_dim=1)
        MultivariateOutput = td.MultivariateNormal(means,torch.diag_embed(torch.exp(variances)))
        return MultivariateOutput

class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x): 
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x) # Normal law with parameters mu and sigma, learned by the encoder
        z = q.rsample() # Sample z using reparametrisation trick
        if isinstance(self.decoder,BernoulliDecoder):
            elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0) 
            # ELBO = Expectation( Distance btw real image and latent distribution (realism) - gap btw latent distr. and prior (we want a standard gaussian)
            # Dans le premier terme on passe par un décodeur pour pouvoir mesurer la distance entre la distribution latente (dimension M) et la distribution de la donnée (celle de x)
        else :
            elbo = torch.mean(self.decoder(z).log_prob(torch.flatten(x,start_dim=1)) - td.kl_divergence(q, self.prior()), dim=0) 

        return elbo
    
    def elbo_mc(self, x, N_iterations=1): # ELBO(x)=Ez∼qϕ(z∣x)   [lnp(x∣z) + lnp(z) − lnqϕ(z∣x)]
        """
        Compute the ELBO for the given batch of data using another formulation.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        N_iterations : int
            Number of iteration in the Monte Carlo process
        """
        Elbo_total = 0
        q = self.encoder(x) # Aggregate Normal posterior with learned parameters from encoder
        for _ in range(N_iterations):
            z = q.rsample() # sample z from the aggregate posterior
            Elbo_total += torch.mean(self.decoder(z).log_prob(x) + self.prior.log_prob(z) - q.log_prob(z))
        return Elbo_total / N_iterations


    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior.sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    

    def aggregated_posterior_sample(self, batch_x):
        # Aggregate posterior from this datapoint :
        q = self.encoder(batch_x)
        # Sample from the aggregate posterior, no need for reparametrisation trick here since we are not doing backpropagation 
        z = q.sample()
        
        return z # z Batch-size tensor of latent data, every data point is represented by a latent vector z sampled from an independant distribution thanks to td.Independant

    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        if isinstance(self.prior, GaussianMixturePrior) | isinstance(self.prior, Flow) :
            return -self.elbo_mc(x) # Monte carlo method 
        else :
            return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device, prior, test_loader = None, validation = False):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    

    if prior == 'GM':
        print("ELBO using gaussian mixture prior")
    elif prior == 'Standard' :
        print("ELBO using standard gaussian prior")
    elif prior == 'Flow' :
        print("ELBO using Flow based prior")

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")
    if validation :
        validation_losses = []
        training_losses = []
    for epoch in range(epochs):
        model.train()
        data_iter = iter(data_loader)
        if validation :
            total_training_elbo = 0
            num_batches = 0
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            if validation :
                total_training_elbo += loss.item()* x.size(0)
                num_batches +=1
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()

        if validation :
            mean_training_elbo = total_training_elbo / len(data_loader.dataset)
            training_losses.append(mean_training_elbo)

                
            model.eval()
            with torch.no_grad():
                total_validation_elbo = 0
                num_batches = 0
                for x in test_loader:
                    x = x[0].to(device)
                    loss = model(x)
                    total_validation_elbo += loss.item() * x.size(0)                
                    num_batches += 1
                mean_validation_elbo = total_validation_elbo / len(test_loader.dataset)
            validation_losses.append(mean_validation_elbo)

    if validation :
        return training_losses, validation_losses

            

def pca_gpu(x, n_components=2):
    """PCA for tensors on GPU"""
   
    mean = torch.mean(x, dim=0)
    x_centered = x - mean
    
    U, S, Vh = torch.linalg.svd(x_centered, full_matrices=False)
    
    components = Vh[:n_components]
    
    projected = torch.mm(x_centered, components.t())
    
    return projected, components


if __name__ == "__main__":
    print("Entering main ...")
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    import matplotlib.pyplot as plt
    from Color_mapping import color_mapping, grid_for_distribution_plot
    from matplotlib.colors import LogNorm, PowerNorm

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='training-curve', choices=['train', 'sample', 'evaluate', 'color-map','training-curve'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='mps', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--dataset', type=str, default='bin', metavar='N', choices = ['bin', 'cont'], help='choice of the MNIST data set version  (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='Standard', choices=['Standard','GM','Flow'], metavar='N',help='Std Gaussian, Gaussian Mixture or Flow based prior (default: %(default)s)')
    parser.add_argument('--validation', type=bool, default=False, metavar='N',help='Whether to compute validation loss (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device


    # Load MNIST as binarized at 'thresshold' and create data loaders
    print("loading train & test sets ...")
    thresshold = 0.5

    if args.dataset == 'bin' :
        mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
        mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    
    # Continuous dataset
    else :
        mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST (' data /' , train = True , download = True ,transform = transforms.Compose ([transforms.ToTensor(),transforms.Lambda ( lambda x : x.squeeze () )]) ),
                                                          batch_size=args.batch_size, shuffle=True)

        mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST (' data /' , train = False , download = True ,transform = transforms.Compose ([transforms.ToTensor(),transforms.Lambda ( lambda x : x.squeeze () )]) ),
                                                          batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    M = args.latent_dim
    if args.prior == 'Standard':
        print("Using standard Gaussian prior")
        prior = GaussianBase(M)
    elif args.prior == 'GM' :
        print("Using Gaussian Mixture prior")
        prior = GaussianMixturePrior(M)
    elif args.prior == 'Flow':
        num_transformations = 10
        num_hidden = floor(2*M/3)  # Dimension of hidden layers for scale and translation nets. For latent_dim = 32, num_hidden = 20
        mask = torch.Tensor([0 if i%2 == 0 else 1 for i in range(M)]) # Checkerboard mask for the permutations in the flow process
        transformations = []
        for i in range(num_transformations): 
            # Flip the mask : permutation layer
            mask = (1 - mask)
            # Scale net for transformation i
            scale_net = nn.Sequential(nn.Linear(M,num_hidden), nn.ReLU(), nn.Linear(num_hidden,M), nn.Tanh())
            # Translation net for transformation i
            translation_net = nn.Sequential(nn.Linear(M,num_hidden), nn.ReLU(), nn.Linear(num_hidden,M))
            transformations.append(MaskedCouplingLayer(scale_net,translation_net,mask))

        prior = Flow(GaussianBase(M),transformations=transformations)

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )
    if args.dataset == 'bin' :
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Unflatten(-1, (28, 28))
        )
    else :
        decoder_net = nn.Sequential(
            nn.Linear(M, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2*784),
            nn.Unflatten(-1, (2, 784))
            #nn.Unflatten(-1, (28, 28))
        )

    # Define VAE model
    if args.dataset == 'bin' :
        decoder = BernoulliDecoder(decoder_net)
    else :
        decoder = MultivariateGaussianDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        if args.dataset == 'bin':
            if args.validation :
                train_losses, validation_losses = train(model, optimizer, mnist_train_loader, args.epochs, args.device, prior = args.prior, test_loader= mnist_test_loader, validation = args.validation)
            else :
                train(model, optimizer, mnist_train_loader, args.epochs, args.device, prior = args.prior, test_loader= mnist_test_loader, validation = args.validation)

        elif args.dataset == 'cont':
            if args.validation : 
                train_losses, validation_losses = train(model, optimizer, mnist_train_loader, args.epochs, args.device, prior = args.prior, test_loader=mnist_test_loader, validation=args.validation)
            else :
                train(model, optimizer, mnist_train_loader, args.epochs, args.device, prior = args.prior, test_loader=mnist_test_loader, validation=args.validation)

        # Save model
        # Save in the directory OUT :

        out_dir = "OUT"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        torch.save(model.state_dict(), os.path.join(out_dir, os.path.basename(args.model)))
        
        if args.validation :
            # Save performances
            losses = {'training': train_losses, 'validation' : validation_losses}
            filename = os.path.join(out_dir, os.path.basename(args.model)[:-2] + 'json')
            with open(filename, "w") as f:
                json.dump(losses, f, indent=4)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), args.samples)

    elif args.mode == 'evaluate' :
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()
        
        total_elbo = 0
        num_batches = 0
        
        print("Calcul de l'ELBO sur le dataset de test...")
        with torch.no_grad():
            for x, label in mnist_test_loader:
                x = x.to(device)
                # We use directly elbo to have a positive value
                batch_elbo = model(x)
                total_elbo += batch_elbo.item()
                num_batches += 1
                #print("x shape : ", x.shape)
                #print("label shape : ",label.shape)
        
        mean_elbo = total_elbo / num_batches # Mean over all batches
        print(f"Averaged ELBO on MNIST test set : {mean_elbo:.4f}", flush=True)
    
    elif args.mode == 'color-map' :
        # Loading mnist test-set
        model.to('cpu')
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()

        Latent_variables = []
        Labels = []
        for batch_x, batch_label in mnist_test_loader :
            if batch_label.shape == torch.Size([128]) :
                # Sample from aggregated posterior
                batch_x = batch_x.to(device)
                z = model.aggregated_posterior_sample(batch_x)
                Latent_variables.append(z)
                Labels.append(batch_label)
        
        Labels = torch.flatten(torch.stack(Labels)).numpy()
        Latent_variables = torch.flatten(torch.stack(Latent_variables),end_dim=-2)

        if args.latent_dim != 2:

            # Perform PCA on the latent variables
            #transformed_latent = pca_gpu(Latent_variables,n_components=2)
            pca = PCA(n_components=2)
            Latent_variables = Latent_variables.cpu().numpy()
            pca.fit(Latent_variables)
            # Project the latent variable on the pca space
            transformed_latent = pca.transform(Latent_variables) 
            # Explained variance
            print("Explained variance from the pca on the latent variables : ", pca.explained_variance_ratio_)
        else : 
            transformed_latent = Latent_variables
        print("shape of transformed latent : ",transformed_latent.shape)
        minx = transformed_latent[:,0].min()
        maxx = transformed_latent[:,0].max()
        miny = transformed_latent[:,1].min()
        maxy = transformed_latent[:,1].max()
        limx = max(-minx,maxx)
        limy = max(-miny,maxy)
        min = min(minx,miny)
        max = max(maxx,maxy)
        prior_grid = grid_for_distribution_plot(dist = model.prior, latent_dim=args.latent_dim, min = min , max = max)

        print(model.prior)

        plt.figure(figsize=(6, 5))
        plt.imshow(prior_grid.detach().numpy(), origin='lower',extent=[min, max, min, max], norm = PowerNorm(gamma=0.2), cmap='viridis')
        plt.colorbar(label='Density')
        plt.xlabel("z1")
        plt.ylabel("z2")
        plt.scatter(transformed_latent[:,0], transformed_latent[:,1],alpha = 0.5,s = 0.5,marker = 'o',c='black')
        plt.show()



        # color_mapping(transformed_latent, Labels)

    elif args.mode == 'training-curve' :
        filename = "Prior_2D_Flow.json"
        with open(filename, "r") as f:
            losses = json.load(f)
        training_losses = np.array(losses['training'])
        validation_losses =  np.array(losses['validation'])
        num_epochs = len(training_losses)
        epochs = np.array([i for i in range(1,num_epochs+1)])
        plt.plot(epochs, training_losses, c = 'orange', label = "training loss")
        plt.plot(epochs, validation_losses, c = 'blue', label = "validation loss")
        plt.title("ELBO training and validation losses vs epochs")
        plt.xlabel("epochs")
        plt.ylabel("ELBO loss")
        plt.legend()
        plt.show()
            


    else : 
        print("Error with mode", flush=True)
        

    
