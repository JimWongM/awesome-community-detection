## Tensor Decomposition

- **1.Coupled Graphs and Tensor Factorization for Recommender Systems and Community Detection (TKDE 2018)**
  - Vassilis N. Ioannidis, Ahmed S. Zamzam, Georgios B. Giannakis, Nicholas D. Sidiropoulos
  - [[Paper]](https://arxiv.org/abs/1809.08353)
  - [[Matlab reference]](https://github.com/bioannidis/Coupled_tensors_graphs)

Joint analysis of data from multiple information repositories facilitates uncovering the underlying structure in heterogeneous datasets. Single and coupled matrix-tensor factorization (CMTF) has been widely used in this context for imputation-based recommendation from ratings, social network, and other user-item data. When this side information is in the form of item-item correlation matrices or graphs, existing CMTF algorithms may fall short. Alleviating current limitations, we introduce a novel model coined coupled graph-tensor factorization (CGTF) that judiciously accounts for graph-related side information. The CGTF model has the potential to overcome practical challenges, such as missing slabs from the tensor and/or missing rows/columns from the correlation matrices. A novel alternating direction method of multipliers (ADMM) is also developed that recovers the nonnegative factors of CGTF. Our algorithm enjoys closed-form updates that result in reduced computational complexity and allow for convergence claims. A novel direction is further explored by employing the interpretable factors to detect graph communities having the tensor as side information. The resulting community detection approach is successful even when some links in the graphs are missing. Results with real data sets corroborate the merits of the proposed methods relative to state-of-the-art competing factorization techniques in providing recommendations and detecting communities.

- **2.Community Detection, Link Prediction, and Layer Interdependence in Multilayer Networks (Physical Review E 2017)**
  - Caterina De Bacco, Eleanor A. Power, Daniel B. Larremore, and Cristopher Moore
  - [[Paper]](https://arxiv.org/abs/1701.01369)
  - [[Python Reference]](https://github.com/cdebacco/MultiTensor)

Complex systems are often characterized by distinct types of interactions between the same entities. These can be described as a multilayer network where each layer represents one type of interaction. These layers may be interdependent in complicated ways, revealing different kinds of structure in the network. In this work we present a generative model, and an efficient expectation-maximization algorithm, which allows us to perform inference tasks such as community detection and link prediction in this setting. Our model assumes overlapping communities that are common between the layers, while allowing these communities to affect each layer in a different way, including arbitrary mixtures of assortative, disassortative, or directed structure. It also gives us a mathematically principled way to define the interdependence between layers, by measuring how much information about one layer helps us predict links in another layer. In particular, this allows us to bundle layers together to compress redundant information, and identify small groups of layers which suffice to predict the remaining layers accurately. We illustrate these findings by analyzing synthetic data and two real multilayer networks, one representing social support relationships among villagers in South India and the other representing shared genetic substrings material between genes of the malaria parasite.

- **3.Overlapping Community Detection via Constrained PARAFAC: A Divide and Conquer Approach (ICDM 2017)**
  - Fatemeh Sheikholeslami and Georgios B. Giannakis 
  - [[Paper]](https://ieeexplore.ieee.org/document/8215485)
  - [[Python Reference]](https://github.com/FatemehSheikholeslami/EgoTen)
  
- **4.Fast Detection of Overlapping Communities via Online Tensor Methods on GPUs (ArXiV 2013)**
  - Furong Huang and Animashree Anandkumar
  - [[Paper]](https://www.semanticscholar.org/paper/Fast-Detection-of-Overlapping-Communities-via-on-Huang-Niranjan/356e6c7eacca6caa94a5a96f41a9c785064f5693)
  - [[C++ Reference]](https://github.com/mapleyustat/Fast-Detection-of-Overlapping-Communities-via-Online-Tensor-Methods)
