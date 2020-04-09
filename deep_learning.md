## Deep Learning

- **1.One2Multi Graph Autoencoder for Multi-view Graph Clustering (WWW 2020)**
  - Shaohua Fan, Xiao Wang, Chuan Shi, Emiao Lu, Ken Lin, Bai Wang
  - [[Paper]](http://www.shichuan.org/doc/83.pdf)
  - [[Python Reference]](https://github.com/googlebaba/WWW2020-O2MAC)

Multi-view graph clustering, which seeks a partition of the graph
with multiple views that often provide more comprehensive yet
complex information, has received considerable attention in recent
years. Although some efforts have been made for multi-view graph
clustering and achieve decent performances, most of them employ
shallow model to deal with the complex relation within multiview graph, which may seriously restrict the capacity for modeling
multi-view graph information. In this paper, we make the first
attempt to employ deep learning technique for attributed multiview graph clustering, and propose a novel task-guided One2Multi
graph autoencoder clustering framework. The One2Multi graph
autoencoder is able to learn node embeddings by employing one
informative graph view and content data to reconstruct multiple
graph views. Hence, the shared feature representation of multiple
graphs can be well captured. Furthermore, a self-training clustering
objective is proposed to iteratively improve the clustering results.
By integrating the self-training and autoencoder’s reconstruction
into a unified framework, our model can jointly optimize the cluster
label assignments and embeddings suitable for graph clustering.
Experiments on real-world attributed multi-view graph datasets
well validate the effectiveness of our model.

- **2.Structural Deep Clustering Network (WWW 2020)**
  - Deyu Bo, Xiao Wang, Chuan Shi, Meiqi Zhu, Emiao Lu, Peng Cui
  - [[Paper]](https://arxiv.org/abs/2002.01633)
  - [[Python Reference]](https://github.com/bdy9527/SDCN)

Clustering is a fundamental task in data analysis. Recently, deep clustering, which derives inspiration primarily from deep learning approaches, achieves state-of-the-art performance and has attracted considerable attention. Current deep clustering methods usually boost the clustering results by means of the powerful representation ability of deep learning, e.g., autoencoder, suggesting that learning an effective representation for clustering is a crucial requirement. The strength of deep clustering methods is to extract the useful representations from the data itself, rather than the structure of data, which receives scarce attention in representation learning. Motivated by the great success of Graph Convolutional Network (GCN) in encoding the graph structure, we propose a Structural Deep Clustering Network (SDCN) to integrate the structural information into deep clustering. Specifically, we design a delivery operator to transfer the representations learned by autoencoder to the corresponding GCN layer, and a dual self-supervised mechanism to unify these two different deep neural architectures and guide the update of the whole model. In this way, the multiple structures of data, from low-order to high-order, are naturally combined with the multiple representations learned by autoencoder. Furthermore, we theoretically analyze the delivery operator, i.e., with the delivery operator, GCN improves the autoencoder-specific representation as a high-order graph regularization constraint and autoencoder helps alleviate the over-smoothing problem in GCN. Through comprehensive experiments, we demonstrate that our propose model can consistently perform better over the state-of-the-art techniques.

- **3.Deep Multi-Graph Clustering via Attentive Cross-Graph Association (WSDM 2020)**
  - Jingchao Ni, Suhang Wang, Yuchen Bian, Xiong Yu and Xiang Zhang 
  - [[Paper]](http://personal.psu.edu/dul262/dmgc.pdf)
  - [[Python Reference]](https://github.com/flyingdoog/DMGC)

Multi-graph clustering aims to improve clustering accuracy by
leveraging information from different domains, which has been
shown to be extremely effective for achieving better clustering
results than single graph based clustering algorithms. Despite the
previous success, existing multi-graph clustering methods mostly
use shallow models, which are incapable to capture the highly
non-linear structures and the complex cluster associations in multigraph, thus result in sub-optimal results. Inspired by the powerful
representation learning capability of neural networks, in this paper,
we propose an end-to-end deep learning model to simultaneously
infer cluster assignments and cluster associations in multi-graph.
Specifically, we use autoencoding networks to learn node embeddings. Meanwhile, we propose a minimum-entropy based clustering
strategy to cluster nodes in the embedding space for each graph.
We introduce two regularizers to leverage both within-graph and
cross-graph dependencies. An attentive mechanism is further developed to learn cross-graph cluster associations. Through extensive
experiments on a variety of datasets, we observe that our method
outperforms state-of-the-art baselines by a large margin.

- **4.Overlapping Community Detection with Graph Neural Networks (MLGWorkShop 2019)**
  - Oleksandr Shchur and Stephan Gunnemann
  - [[Paper]](http://www.kdd.in.tum.de/research/nocd/)
  - [[Python Reference]](https://github.com/shchur/overlapping-community-detection)
  - [[Python]](https://github.com/EthanNing/Exp-GAE-model)

- **5.Supervised Community Detection with Line Graph Neural Networks (ICLR 2019)**
  - Zhengdao Chen, Xiang Li, and Joan Bruna
  - [[Paper]](https://arxiv.org/abs/1705.08415)
  - [[LUA Reference]](https://github.com/joanbruna/GNN_community)
  - [[Python Reference]](https://github.com/afansi/multiscalegnn)
  - [[Python]](https://github.com/zhengdao-chen/GNN4CD)

We study data-driven methods for community detection on graphs, an inverse problem that is typically solved in terms of the spectrum of certain operators or via posterior inference under certain probabilistic graphical models. Focusing on random graph families such as the stochastic block model, recent research has unified both approaches and identified both statistical and computational signal-to-noise detection thresholds. This graph inference task can be recast as a node-wise graph classification problem, and, as such, computational detection thresholds can be translated in terms of learning within appropriate models. We present a novel family of Graph Neural Networks (GNNs) and show that they can reach those detection thresholds in a purely data-driven manner without access to the underlying generative models, and even improve upon current computational thresholds in hard regimes. For that purpose, we propose to augment GNNs with the non-backtracking operator, defined on the line graph of edge adjacencies. We also perform the first analysis of optimization landscape on using GNNs to solve community detection problems, demonstrating that under certain simplifications and assumptions, the loss value at the local minima is close to the loss value at the global minimum/minima. Finally, the resulting model is also tested on real datasets, performing significantly better than previous models.

- **6.CommunityGAN: Community Detection with Generative Adversarial Nets (ArXiv 2019)**
  - Yuting Jia, Qinqin Zhang, Weinan Zhang, Xinbing Wang
  - [[Paper]](https://arxiv.org/abs/1901.06631)
  - [[Python Reference]](https://github.com/SamJia/CommunityGAN)

Community detection refers to the task of discovering groups of vertices sharing similar properties or functions so as to understand the network data. With the recent development of deep learning, graph representation learning techniques are also utilized for community detection. However, the communities can only be inferred by applying clustering algorithms based on learned vertex embeddings. These general cluster algorithms like K-means and Gaussian Mixture Model cannot output much overlapped communities, which have been proved to be very common in many real-world networks. In this paper, we propose CommunityGAN, a novel community detection framework that jointly solves overlapping community detection and graph representation learning. First, unlike the embedding of conventional graph representation learning algorithms where the vector entry values have no specific meanings, the embedding of CommunityGAN indicates the membership strength of vertices to communities. Second, a specifically designed Generative Adversarial Net (GAN) is adopted to optimize such embedding. Through the minimax competition between the motif-level generator and discriminator, both of them can alternatively and iteratively boost their performance and finally output a better community structure. Extensive experiments on synthetic data and real-world tasks demonstrate that CommunityGAN achieves substantial community detection performance gains over the state-of-the-art methods.

- **7.An Adaptive Graph Learning Method Based on Dual Data Representations for Clustering (Pattern Recognition 2018)**
  - Tianchi Liu, Chamara Kasun, Liyanaarachchi Lekamalage Guang-Bin Huang, and Zhiping Lin
  - [[Paper]](https://www.sciencedirect.com/science/article/pii/S0031320317304880)
  - [[Matlab Reference]](https://github.com/liut0012/ELM-CLR)

Adaptive graph learning methods for clustering, which adjust a data similarity matrix while taking into account its clustering capability, have drawn increasing attention in recent years due to their promising clustering performance. Existing adaptive graph learning methods are based on either original data or linearly projected data and thus rely on the assumption that either representation is a good indicator of the underlying data structure. However, this assumption is sometimes not met in high dimensional data. Studies have shown that high-dimensional data in many problems tend to lie on an embedded nonlinear manifold structure. Motivated by this observation, in this paper, we develop dual data representations, i.e., original data and a nonlinear embedding of the data obtained via an Extreme Learning Machine (ELM)-based neural network, and propose to use them as the more reliable basis for graph learning. The resulting algorithm based on ELM and Constrained Laplacian Rank (ELM-CLR) further improves the clustering capability and robustness, while retaining the advantages of adaptive graph learning, such as not requiring any post-processing to extract cluster indicators. The empirical study shows that the proposed algorithm outperforms the state-of-the-art graph-based clustering methods on a broad range of benchmark datasets.

- **8.Improving the Efficiency and Effectiveness of Community Detection via Prior-Induced Equivalent Super-Network (Scientific Reports 2017)**
  - Liang Yang, Di Jin, Dongxiao He, Huazhu Fu, Xiaochun Cao, and Francoise Fogelman-Soulie
  - [[Paper]](http://yangliang.github.io/pdf/sr17.pdf)
  - [[Python Reference]](http://yangliang.github.io/code/SUPER.zip)

Due to the importance of community structure in understanding network and a surge of interest
aroused on community detectability, how to improve the community identification performance with
pairwise prior information becomes a hot topic. However, most existing semi-supervised community
detection algorithms only focus on improving the accuracy but ignore the impacts of priors on speeding
detection. Besides, they always require to tune additional parameters and cannot guarantee pairwise
constraints. To address these drawbacks, we propose a general, high-speed, effective and parameterfree semi-supervised community detection framework. By constructing the indivisible super-nodes
according to the connected subgraph of the must-link constraints and by forming the weighted superedge based on network topology and cannot-link constraints, our new framework transforms the
original network into an equivalent but much smaller Super-Network. Super-Network perfectly ensures
the must-link constraints and effectively encodes cannot-link constraints. Furthermore, the time
complexity of super-network construction process is linear in the original network size, which makes it
efficient. Meanwhile, since the constructed super-network is much smaller than the original one, any
existing community detection algorithm is much faster when using our framework. Besides, the overall
process will not introduce any additional parameters, making it more practical.

- **9.MGAE: Marginalized Graph Autoencoder for Graph Clustering (CIKM 2017)**
  - Chun Wang, Shirui Pan, Guodong Long, Xingquabn Zhu, and Jing Jiang
  - [[Paper]](https://dl.acm.org/citation.cfm?id=3132967)
  - [[Matlab Reference]](https://github.com/FakeTibbers/MGAE)

Graph clustering aims to discovercommunity structures in networks, the task being fundamentally challenging mainly because the topology structure and the content of the graphs are difficult to represent for clustering analysis. Recently, graph clustering has moved from traditional shallow methods to deep learning approaches, thanks to the unique feature representation learning capability of deep learning. However, existing deep approaches for graph clustering can only exploit the structure information, while ignoring the content information associated with the nodes in a graph. In this paper, we propose a novel marginalized graph autoencoder (MGAE) algorithm for graph clustering. The key innovation of MGAE is that it advances the autoencoder to the graph domain, so graph representation learning can be carried out not only in a purely unsupervised setting by leveraging structure and content information, it can also be stacked in a deep fashion to learn effective representation. From a technical viewpoint, we propose a marginalized graph convolutional network to corrupt network node content, allowing node content to interact with network features, and marginalizes the corrupted features in a graph autoencoder context to learn graph feature representations. The learned features are fed into the spectral clustering algorithm for graph clustering. Experimental results on benchmark datasets demonstrate the superior performance of MGAE, compared to numerous baselines.

- **10.Graph Clustering with Dynamic Embedding (Arxiv 2017)**
  - Carl Yang, Mengxiong Liu, Zongyi Wang, Liyuan Liu, Jiawei Han
  - [[Paper]](https://arxiv.org/abs/1712.08249)
  - [[Python Reference]](https://github.com/yangji9181/GRACE)

Graph clustering (or community detection) has long drawn enormous attention from the research on web mining and information networks. Recent literature on this topic has reached a consensus that node contents and link structures should be integrated for reliable graph clustering, especially in an unsupervised setting. However, existing methods based on shallow models often suffer from content noise and sparsity. In this work, we propose to utilize deep embedding for graph clustering, motivated by the well-recognized power of neural networks in learning intrinsic content representations. Upon that, we capture the dynamic nature of networks through the principle of influence propagation and calculate the dynamic network embedding. Network clusters are then detected based on the stable state of such an embedding. Unlike most existing embedding methods that are task-agnostic, we simultaneously solve for the underlying node representations and the optimal clustering assignments in an end-to-end manner. To provide more insight, we theoretically analyze our interpretation of network clusters and find its underlying connections with two widely applied approaches for network modeling. Extensive experimental results on six real-world datasets including both social networks and citation networks demonstrate the superiority of our proposed model over the state-of-the-art.

- **11.Modularity based Community Detection with Deep Learning (IJCAI 2016)**
  - Liang Yang, Xiaochun Cao, Dongxiao He, Chuan Wang, Xiao Wang, and Weixiong Zhan
  - [[Paper]](http://yangliang.github.io/pdf/ijcai16.pdf)
  - [[Python Reference]](http://yangliang.github.io/code/DC.zip)

Identification of module or community structures
is important for characterizing and understanding
complex systems. While designed with different objectives, i.e., stochastic models for regeneration and
modularity maximization models for discrimination,
both these two types of model look for low-rank
embedding to best represent and reconstruct network topology. However, the mapping through such
embedding is linear, whereas real networks have various nonlinear features, making these models less
effective in practice. Inspired by the strong representation power of deep neural networks, we propose a
novel nonlinear reconstruction method by adopting
deep neural networks for representation. We then
extend the method to a semi-supervised community
detection algorithm by incorporating pairwise constraints among graph nodes. Extensive experimental
results on synthetic and real networks show that
the new methods are effective, outperforming most
state-of-the-art methods for community detection.

- **12.Learning Deep Representations for Graph Clustering (AAAI 2014)**
  - Fei Tian, Bin Gao, Qing Cui, Enhong Chen, and Tie-Yan Liu
  - [[Paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8527)
  - [[Python Reference]](https://github.com/quinngroup/deep-representations-clustering)
  - [[Python Alternative]](https://github.com/zepx/graphencoder)

Recently deep learning has been successfully adopted in many applications such as speech recognition and image classification. In this work, we explore the possibility of employing deep learning in graph clustering. We propose a simple method, which first learns a nonlinear embedding of the original graph by stacked autoencoder, and then runs $k$-means algorithm on the embedding to obtain the clustering result. We show that this simple method has solid theoretical foundation, due to the similarity between autoencoder and spectral clustering in terms of what they actually optimize. Then, we demonstrate that the proposed method is more efficient and flexible than spectral clustering. First, the computational complexity of autoencoder is much lower than spectral clustering: the former can be linear to the number of nodes in a sparse graph while the latter is super quadratic due to eigenvalue decomposition. Second, when additional sparsity constraint is imposed, we can simply employ the sparse autoencoder developed in the literature of deep learning; however, it is non-straightforward to implement a sparse spectral method. The experimental results on various graph datasets show that the proposed method significantly outperforms conventional spectral clustering which clearly indicates the effectiveness of deep learning in graph clustering.
