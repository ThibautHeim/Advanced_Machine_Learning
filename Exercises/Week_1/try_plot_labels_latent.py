
from sklearn.decomposition import PCA
import numpy as np

def aggregated_posterior_sample(self, data_points):
        # Tensor pytorch to contain the latent sample data
        # For every latent tensor, the associated label i.e. in : 0, 1, ..., 9
        Latent_data = []
        Labels = []
        
        # Isolate data points from the data
        for x, label in data_points :
            # x : 28*28
            # label : 1
            q = self.encoder(x)
            # Sample from the aggregated posterior using the reparametrisation trick
            z = q.rsample()
            n = z.shape[0]
            Latent_data.append(z) # A MODIFIER
            # For every sampled data from this data point, there is an associated label single_label
            for i in range(n):
                Labels.append(label)
                Latent_data.append(z[i])

        pca = PCA(n_components=2)
        pca.fit(np.array(Latent_data))

        return pca, Labels


        