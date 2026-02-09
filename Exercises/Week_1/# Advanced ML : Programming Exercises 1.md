# Advanced ML : Programming Exercises - Week 1

## Exercise 1.4

> In this first exercise, you should just inspect the code in vae_bernoulli.py.
Answer the following questions:

>* How is the reparametrisation trick handled in the code?

The reparametrisation trick is handled when sampling z from the distribution $q(x| \phi )$ only noted $q$ in the code and learned using Gaussian encoder class.  
This sampling is done in the $elbo$ function within the $VAE$ class


``` 
def elbo(self, x):
    ...
    q = self.encoder(x)
    z = q.rsample() 
    ...
    return elbo
```


>* Consider the implementation of the ELBO.  
What is the dimension of `self.decoder(
z).log_prob(x)` and of `td.kl_divergence(q, self.prior.distribution)`?

* `self.decoder(z).log_prob(x)` is of dimension $B$, the batch size = number of images = $128$. It is the log product Bernouilli likelihood.
* `td.kl_divergence(q, self.prior.distribution)` is of the same dimension. It is the KL divergence between the latent distribution $q$ and the prior standard Gaussian distribution $p(z)$.

>* The implementation of the prior, encoder and decoder classes all make use of
td.Independent. What does this do?

`td.Independant` enables to consider every image of the batch as a unique event. This allows to compute directly the log likelihood of the whole image using `.log_prob(x)` instead of considering every pixel independantly.

>* What is the purpose using the function torch.chunk in GaussianEncoder.forward?

`torch.chunk(self.encoder_net(x), 2, dim=-1)` enables to split the encoded input x into two chunks over the last dimension (columns)
in order to have a column of $means$ and a columns of $\log(stds)$ for the approximated latent distribution $q$.

## Exercise 1.5

> Add the following functionally to the implementation (vae_bernoulli.py) of
the VAE with Bernoulli output distributions:

> * Evaluate the ELBO on the binarised MNIST test set.

In order to evaluate the ELBO on the MINST dataset, we added another possible argument for the `mode` mandatory argument : `evaluate`:

```
if args.mode == 'train':
    ...

elif args.mode == 'sample':
    ...

elif args.mode == 'evaluate' :
    model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
    model.eval()
    
    total_elbo = 0
    num_batches = 0
    
    print("Calcul de l'ELBO sur le dataset de test...")
    with torch.no_grad():
        for x, _ in mnist_test_loader:
            x = x.to(device)
            # We use directly elbo to have a positive value
            batch_elbo = model.elbo(x)
            total_elbo += batch_elbo.item()
            num_batches += 1
    
    mean_elbo = total_elbo / num_batches # Mean over all batches
    print(f"ELBO moyen sur le dataset MNIST test : {mean_elbo:.4f}")
```

In the end we end up with a value of $\approx 95$ for the elbo MNIST test set.


> * Plot samples from the approximate posterior and colour them by their correct class
label for each datapoint in the test set (i.e., samples from the aggregate posterior).
Implement it such that you, for latent dimensions larger than two, M > 2, do
PCA and project the sample onto the first two principal components (e.g., using
scikit-learn).



