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
        self.categorical_weights = nn.Parameter(categorical_weights, requires_grad=False)
        if locs is not None :
            self.locs = nn.Parameter(locs, requires_grad=False) # tensor of nb_gaussians loc vectors
        else :
            self.locs = nn.Parameter(torch.zeros(10,self.M), requires_grad=False)
        if scales is not None :    
            self.scales = nn.Parameter(scales,requires_grad=False) # tensor of nb_gaussians scale vectors
        else :
            self.scales = nn.Parameter(torch.ones(10,self.M), requires_grad=False)
        self.nb_gaussians = len(self.locs)
        # List of Gaussian distributions 
        self.gaussians = td.Independent(td.Normal(self.locs,self.scales),1) # Batch of nb_gaussians (ex 10) normal diagonal distributions parametrized by the vectors self.locs[i] and self.means[i]
        self.categorical = td.Categorical(categorical_weights)
        self.prior = td.MixtureSameFamily(mixture_distribution=self.categorical, component_distribution=self.gaussians)
    
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



class GaussianPrior(nn.Module): # p(z) = N(0, I)
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
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
        q = self.encoder(x) # Loi normale de paramètres mu et sigmas appris par encoder
        z = q.rsample() # Sample z en utilisant reparametrisation trick
        if isinstance(self.decoder,BernoulliDecoder):
            elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0) 
            # ELBO = Espérance( Distance entre l'image réelle et la distribution latente (réalisme) - écart entre la distribution latente et la prior (on veut une gaussienne centrée réduite) )
            # Dans le premier terme on passe par un décodeur pour pouvoir mesurer la distance entre la distribution latente (dimension M) et la distribution de la donnée (celle de x)
        else :
            elbo = torch.mean(self.decoder(z).log_prob(torch.flatten(x,start_dim=1)) - td.kl_divergence(q, self.prior()), dim=0) 
        return elbo
    
    def elbo_mc(self, x, N_iterations=10): # ELBO(x)=Ez∼qϕ(z∣x)   [lnp(x∣z) + lnp(z) − lnqϕ(z∣x)]
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
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    

    def aggregated_posterior_sample(self, batch_x):
        # Aggregate posterior from this datapoint :
        q = self.encoder(batch_x)
        # Sample from the aggregate posterior, no need for reparametrisation trick here since we are not doing backpropagation 
        z = q.sample()
        
        return z # z Batch-size tensor of latent data, every data point is represented by a latent vector z sampled from an independant distribution thanks to td.Independant

    
    def forward(self, x, mc):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        if mc == 'Y':
            return -self.elbo_mc(x)
        else :
            return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device, mc):
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
    model.train()

    if mc == 'Y':
        print("ELBO using gaussian mixture prior")
    else :
        print("ELBO using standard gaussian prior")

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x, mc)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()

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
    from torchvision.utils import save_image, make_grid
    import glob
    import matplotlib.pyplot as plt
    from Color_mapping import color_mapping

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'evaluate', 'color-map'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='mps', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--dataset', type=str, default='bin', metavar='N', choices = ['bin', 'cont'], help='choice of the MNIST data set version  (default: %(default)s)')
    parser.add_argument('--mc', type=str, default='Y', choices=['Y','N'], metavar='N',help='True : MoG prior, False : Gaussian prior (default: %(default)s)')

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
    if args.mc == 'N':
        print("Using standard Gaussian prior")
        prior = GaussianPrior(M)
    else :
        print("Using Gaussian Mixture prior")
        prior = GaussianMixturePrior(M)

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
        )

    # Define VAE model
    #decoder = BernoulliDecoder(decoder_net)
    decoder = MultivariateGaussianDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        if args.dataset == 'bin':
            train(model, optimizer, mnist_train_loader, args.epochs, args.device, mc = args.mc)
        elif args.dataset == 'cont':
            train(model, optimizer, mnist_train_loader, args.epochs, args.device, mc = args.mc)

        # Save model
        torch.save(model.state_dict(), args.model)

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
                batch_elbo = model.elbo(x)
                total_elbo += batch_elbo.item()
                num_batches += 1
                #print("x shape : ", x.shape)
                #print("label shape : ",label.shape)
        
        mean_elbo = total_elbo / num_batches # Mean over all batches
        print(f"Averaged ELBO on MNIST test set : {mean_elbo:.4f}", flush=True)
    
    elif args.mode == "color-map" :
        # Loading mnist test-set
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

        # Perform PCA on the latent variables
        #transformed_latent = pca_gpu(Latent_variables,n_components=2)
        pca = PCA(n_components=2)
        Latent_variables = Latent_variables.cpu().numpy()
        pca.fit(Latent_variables)
        # Project the latent variable on the pca space
        transformed_latent = pca.transform(Latent_variables) 
        # Explained variance
        print("Explained variance from the pca on the latent variables : ", pca.explained_variance_ratio_)

        color_mapping(transformed_latent, Labels)
        
        # + PCA
        #pca = model.aggregated_posterior_sample(mnist_test_loader.flatten())
        #print("\nExplained variance from latent space : ", pca.explained_variance_ratio_,"\n")

    else : 
        print("On ne rentre dans aucun if/elif", flush=True)
        

    
