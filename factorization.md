## Matrix Factorization

- **1.Gromov-Wasserstein Factorization Models for Graph Clustering (AAAI 2020)**
  - Hongteng Xu
  - [[Paper]](https://arxiv.org/abs/1911.08530)
  - [[Python Reference]](https://github.com/HongtengXu/Relational-Factorization-Model)

We propose a new nonlinear factorization model for graphs that are with topological structures, and optionally, node attributes. This model is based on a pseudometric called Gromov-Wasserstein (GW) discrepancy, which compares graphs in a relational way. It estimates observed graphs as GW barycenters constructed by a set of atoms with different weights. By minimizing the GW discrepancy between each observed graph and its GW barycenter-based estimation, we learn the atoms and their weights associated with the observed graphs. The model achieves a novel and flexible factorization mechanism under GW discrepancy, in which both the observed graphs and the learnable atoms can be unaligned and with different sizes. We design an effective approximate algorithm for learning this Gromov-Wasserstein factorization (GWF) model, unrolling loopy computations as stacked modules and computing gradients with backpropagation. The stacked modules can be with two different architectures, which correspond to the proximal point algorithm (PPA) and Bregman alternating direction method of multipliers (BADMM), respectively. Experiments show that our model obtains encouraging results on clustering graphs.

- **2.Graph Embedding with Self-Clustering (ASONAM 2019)**
  - Benedek Rozemberczki, Ryan Davies, Rik Sarkar, and Charles Sutton
  - [[Paper]](https://arxiv.org/abs/1802.03997)
  - [[Python Reference]](https://github.com/benedekrozemberczki/GEMSEC)

Modern graph embedding procedures can efficiently process graphs with millions of nodes. In this paper, we propose GEMSEC -- a graph embedding algorithm which learns a clustering of the nodes simultaneously with computing their embedding. GEMSEC is a general extension of earlier work in the domain of sequence-based graph embedding. GEMSEC places nodes in an abstract feature space where the vertex features minimize the negative log-likelihood of preserving sampled vertex neighborhoods, and it incorporates known social network properties through a machine learning regularization. We present two new social network datasets and show that by simultaneously considering the embedding and clustering problems with respect to social properties, GEMSEC extracts high-quality clusters competitive with or superior to other community detection algorithms. In experiments, the method is found to be computationally efficient and robust to the choice of hyperparameters.

- **3.Consistency Meets Inconsistency: A Unified Graph Learning Framework for Multi-view Clustering (ICDM 2019)**
  - Youwei Liang, Dong Huang, and Chang-Dong Wang
  - [[Paper]](https://youweiliang.github.io/files/consistent_graph_learning.pdf)
  - [[Matlab Reference]](https://github.com/youweiliang/ConsistentGraphLearning)

Abstract—Graph Learning has emerged as a promising technique for multi-view clustering, and has recently attracted lots of
attention due to its capability of adaptively learning a unified and
probably better graph from multiple views. However, the existing
multi-view graph learning methods mostly focus on the multiview consistency, but neglect the potential multi-view inconsistency (which may be incurred by noise, corruptions, or view-specific
characteristics). To address this, this paper presents a new graph
learning-based multi-view clustering approach, which for the first
time, to our knowledge, simultaneously and explicitly formulates
the multi-view consistency and the multi-view inconsistency in
a unified optimization model. To solve this model, a new alternating optimization scheme is designed, where the consistent and
inconsistent parts of each single-view graph as well as the unified
graph that fuses the consistent parts of all views can be iteratively
learned. It is noteworthy that our multi-view graph learning
model is applicable to both similarity graphs and dissimilarity
graphs, leading to two graph fusion-based variants, namely,
distance (dissimilarity) graph fusion and similarity graph fusion.
Experiments on various multi-view datasets demonstrate the superiority of our approach. The MATLAB source code is available
at https://github.com/youweiliang/ConsistentGraphLearning.

- **4.GMC: Graph-based Multi-view Clustering (TKDE 2019)**
  - Hao Wang, Yan Yang, Bing Liu
  - [[Paper]](https://www.researchgate.net/publication/331602096_GMC_Graph-based_Multi-view_Clustering)
  - [[Matlab Reference]](https://github.com/cshaowang/gmc)

Multi-view graph-based clustering aims to provide clustering solutions to multi-view data. However, most existing methods do not give sufficient consideration to weights of different views and require an additional clustering step to produce the final clusters. They also usually optimize their objectives based on fixed graph similarity matrices of all views. In this paper, we propose a general Graph-based Multi-view Clustering (GMC) to tackle these problems. GMC takes the data graph matrices of all views and fuses them to generate a unified matrix. The unified matrix in turn improves the data graph matrix of each view, and also gives the final clusters directly. The key novelty of GMC is its learning method, which can help the learning of each view graph matrix and the learning of the unified matrix in a mutual reinforcement manner. A novel multi-view fusion technique can automatically weight each data graph matrix to derive the unified matrix. A rank constraint without introducing a tuning parameter is also imposed on the Laplacian matrix of the unified matrix, which helps partition the data points naturally into the required number of clusters. An alternating iterative optimization algorithm is presented to optimize the objective function. Experimental results demonstrate that the proposed method outperforms state-of-the-art baselines markedly.

- **5.Embedding-based Silhouette Community Detection (Arxiv 2019)**
  - Blaž Škrlj, Jan Kralj, Nada Lavrač
  - [[Paper]](https://arxiv.org/abs/1908.02556)
  - [[Python Reference]](https://arxiv.org/abs/1908.02556)

Mining complex data in the form of networks is of increasing interest in many scientific disciplines. Network communities correspond to densely connected subnetworks, and often represent key functional parts of real-world systems. In this work, we propose Silhouette Community Detection (SCD), an approach for detecting communities, based on clustering of network node embeddings, i.e. real valued representations of nodes derived from their neighborhoods. We investigate the performance of the proposed SCD approach on 234 synthetic networks, as well as on a real-life social network. Even though SCD is not based on any form of modularity optimization, it performs comparably or better than state-of-the-art community detection algorithms, such as the InfoMap and Louvain algorithms. Further, we demonstrate how SCD's outputs can be used along with domain ontologies in semantic subgroup discovery, yielding human-understandable explanations of communities detected in a real-life protein interaction network. Being embedding-based, SCD is widely applicable and can be tested out-of-the-box as part of many existing network learning and exploration pipelines.

- **6.Knowledge Graph Enhanced Community Detection and Characterization (WSDM 2019)**
  - Shreyansh Bhatt, Swati Padhee, Amit Sheth, Keke Chen ,Valerie Shalin, Derek Doran, and Brandon Minnery 
  - [[Paper]](https://dl.acm.org/authorize.cfm?key=N676882)
  - [[Java Reference]](https://github.com/shreyanshbhatt/KnowledgeGraph_in_CommunityDetection)

Recent studies show that by combining network topology and node attributes, we can better understand community structures in complex networks. However, existing algorithms do not explore "contextually" similar node attribute values, and therefore may miss communities defined with abstract concepts. We propose a community detection and characterization algorithm that incorporates the contextual information of node attributes described by multiple domain-specific hierarchical concept graphs. The core problem is to find the context that can best summarize the nodes in communities, while also discovering communities aligned with the context summarizing communities. We formulate the two intertwined problems, optimal community-context computation, and community discovery, with a coordinate-ascent based algorithm that iteratively updates the nodes' community label assignment with a community-context and computes the best context summarizing nodes of each community. Our unique contributions include (1) a composite metric on Informativeness and Purity criteria in searching for the best context summarizing nodes of a community; (2) a node similarity measure that incorporates the context-level similarity on multiple node attributes; and (3) an integrated algorithm that drives community structure discovery by appropriately weighing edges. Experimental results on public datasets show nearly 20 percent improvement on F-measure and Jaccard for discovering underlying community structure over the current state-of-the-art of community detection methods. Community structure characterization was also accurate to find appropriate community types for four datasets.

- **7.Discrete Optimal Graph Clustering (IEEE Cybernetics 2019)**
  - Yudong Han, Lei Zhu, Zhiyong Cheng, Jingjing Li, Xiaobai Liu
  - [[Paper]](https://arxiv.org/abs/1904.11266)
  - [[Matlab Reference]](https://github.com/christinecui/DOGC)

Graph based clustering is one of the major clustering methods. Most of it work in three separate steps: similarity graph construction, clustering label relaxing and label discretization with k-means. Such common practice has three disadvantages: 1) the predefined similarity graph is often fixed and may not be optimal for the subsequent clustering. 2) the relaxing process of cluster labels may cause significant information loss. 3) label discretization may deviate from the real clustering result since k-means is sensitive to the initialization of cluster centroids. To tackle these problems, in this paper, we propose an effective discrete optimal graph clustering (DOGC) framework. A structured similarity graph that is theoretically optimal for clustering performance is adaptively learned with a guidance of reasonable rank constraint. Besides, to avoid the information loss, we explicitly enforce a discrete transformation on the intermediate continuous label, which derives a tractable optimization problem with discrete solution. Further, to compensate the unreliability of the learned labels and enhance the clustering accuracy, we design an adaptive robust module that learns prediction function for the unseen data based on the learned discrete cluster labels. Finally, an iterative optimization strategy guaranteed with convergence is developed to directly solve the clustering results. Extensive experiments conducted on both real and synthetic datasets demonstrate the superiority of our proposed methods compared with several state-of-the-art clustering approaches.

- **8.Total Variation Based Community Detection Using a Nonlinear Optimization Approach (Arxiv 2019)**
  - Andrea Cristofari, Francesco Rinaldi, Francesco Tudisco
  - [[Paper]](https://arxiv.org/abs/1907.08048)
  - [[C++ Reference]](https://github.com/acristofari/fast-atvo)

Maximizing the modularity of a network is a successful tool to identify an important community of nodes. However, this combinatorial optimization problem is known to be NP-hard. Inspired by recent nonlinear modularity eigenvector approaches, we introduce the modularity total variation TVQ and show that its box-constrained global maximum coincides with the maximum of the original discrete modularity function. Thus we describe a new nonlinear optimization approach to solve the equivalent problem leading to a community detection strategy based on TVQ. The proposed approach relies on the use of a fast first-order method that embeds a tailored active-set strategy. We report extensive numerical comparisons with standard matrix-based approaches and the Generalized Ratio DCA approach for nonlinear modularity eigenvectors, showing that our new method compares favourably with state-of-the-art alternatives. Our software is available upon request.

- **9.vGraph: A Generative Model for Joint Community Detection and Node Representation Learning (NeurIPS 2019)**
  - Fan-Yun Sun, Meng Qu, Jordan Hoffmann, Chin-Wei Huang, Jian Tang
  - [[Paper]](https://arxiv.org/abs/1906.07159)
  - [[Python Reference]](https://github.com/aniket-agarwal1999/vGraph-Pytorch)

This paper focuses on two fundamental tasks of graph analysis: community detection and node representation learning, which capture the global and local structures of graphs, respectively. In the current literature, these two tasks are usually independently studied while they are actually highly correlated. We propose a probabilistic generative model called vGraph to learn community membership and node representation collaboratively. Specifically, we assume that each node can be represented as a mixture of communities, and each community is defined as a multinomial distribution over nodes. Both the mixing coefficients and the community distribution are parameterized by the low-dimensional representations of the nodes and communities. We designed an effective variational inference algorithm which regularizes the community membership of neighboring nodes to be similar in the latent space. Experimental results on multiple real-world graphs show that vGraph is very effective in both community detection and node representation learning, outperforming many competitive baselines in both tasks. We show that the framework of vGraph is quite flexible and can be easily extended to detect hierarchical communities.

- **10.Deep Autoencoder-like Nonnegative Matrix Factorization for Community Detection (CIKM 2018)**
  - Fanghua Ye, Chuan Chen, and Zibin Zheng
  - [[Paper]](https://smartyfh.com/Documents/18DANMF.pdf)
  - [[Python Reference]](https://github.com/benedekrozemberczki/DANMF)
  - [[Matlab Reference]](https://github.com/smartyfh/DANMF)

Community structure is ubiquitous in real-world complex
networks. The task of community detection over these networks is of paramount importance in a variety of applications.
Recently, nonnegative matrix factorization (NMF) has been
widely adopted for community detection due to its great interpretability and its natural fitness for capturing the community membership of nodes. However, the existing NMF-based
community detection approaches are shallow methods. They
learn the community assignment by mapping the original
network to the community membership space directly. Considering the complicated and diversified topology structures
of real-world networks, it is highly possible that the mapping
between the original network and the community membership space contains rather complex hierarchical information,
which cannot be interpreted by classic shallow NMF-based
approaches. Inspired by the unique feature representation
learning capability of deep autoencoder, we propose a novel
model, named Deep Autoencoder-like NMF (DANMF), for
community detection. Similar to deep autoencoder, DANMF
consists of an encoder component and a decoder component.
This architecture empowers DANMF to learn the hierarchical
mappings between the original network and the final community assignment with implicit low-to-high level hidden
attributes of the original network learnt in the intermediate
layers. Thus, DANMF should be better suited to the community detection task. Extensive experiments on benchmark
datasets demonstrate that DANMF can achieve better performance than the state-of-the-art NMF-based community
detection approaches.

- **11.Adaptive Community Detection Incorporating Topology and Content in Social Networks (Knowledge-Based Systems 2018)**
  - Qin Meng, Jin Di, Lei Kai, Bogdan Gabrys, Katarzyna, Musial-Gabrys
  - [[Paper]](https://www.sciencedirect.com/science/article/pii/S0950705118303885?dgcid=coauthor)
  - [[Matlab Reference]](https://github.com/KuroginQin/ASCD)

In social network analysis, community detection is a basic step to understand the structure and function of networks. Some conventional community detection methods may have limited performance because they merely focus on the networks’ topological structure. Besides topology, content information is another significant aspect of social networks. Although some state-of-the-art methods started to combine these two aspects of information for the sake of the improvement of community partitioning, they often assume that topology and content carry similar information. In fact, for some examples of social networks, the hidden characteristics of content may unexpectedly mismatch with topology. To better cope with such situations, we introduce a novel community detection method under the framework of non-negative matrix factorization (NMF). Our proposed method integrates topology as well as content of networks and has an adaptive parameter (with two variations) to effectively control the contribution of content with respect to the identified mismatch degree. Based on the disjoint community partition result, we also introduce an additional overlapping community discovery algorithm, so that our new method can meet the application requirements of both disjoint and overlapping community detection. The case study using real social networks shows that our new method can simultaneously obtain the community structures and their corresponding semantic description, which is helpful to understand the semantics of communities. Related performance evaluations on both artificial and real networks further indicate that our method outperforms some state-of-the-art methods while exhibiting more robust behavior when the mismatch between topology and content is observed.

- **12.Learning Latent Factors for Community Identification and Summarization (IEEE Access 2018)**
  - Tiantian He, Lun Hu, Keith C. C. Chan, and Pengwei Hu 
  - [[Paper]](https://ieeexplore.ieee.org/document/8374421)
  - [[Executable Reference]](https://github.com/he-tiantian/LFCIS)

Network communities, which are also known as network clusters, are typical latent structures in network data. Vertices in each of these communities tend to interact more and share similar features with each other. Community identification and feature summarization are significant tasks of network analytics. To perform either of the two tasks, there have been several approaches proposed, taking into the consideration of different categories of information carried by the network, e.g., edge structure, node attributes, or both aforementioned. But few of them are able to discover communities and summarize their features simultaneously. To address this challenge, we propose a novel latent factor model for community identification and summarization (LFCIS). To perform the task, the LFCIS first formulates an objective function that evaluating the overall clustering quality taking into the consideration of both edge topology and node features in the network. In the objective function, the LFCIS also adopts an effective component that ensures those vertices sharing with both similar local structures and features to be located into the same clusters. To identify the optimal cluster membership for each vertex, a convergent algorithm for updating the variables in the objective function is derived and used by LFCIS. The LFCIS has been tested with six sets of network data, including synthetic and real networks, and compared with several state-of-the-art approaches. The experimental results show that the LFCIS outperforms most of the prevalent approaches to community discovery in social networks, and the LFCIS is able to identify the latent features that may characterize those discovered communities.

- **13.Bayesian Robust Attributed Graph Clustering: Joint Learning of Partial Anomalies and Group Structure (AAAI 2018)**
  - Aleksandar Bojchevski and Stephan Günnemann
  - [[Paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16363/16542)
  - [[Python Reference]](https://github.com/abojchevski/paican)

We study the problem of robust attributed graph clustering.
In real data, the clustering structure is often obfuscated due to
anomalies or corruptions. While robust methods have been recently introduced that handle anomalies as part of the clustering process, they all fail to account for one core aspect: Since
attributed graphs consist of two views (network structure and
attributes) anomalies might materialize only partially, i.e. instances might be corrupted in one view but perfectly fit in
the other. In this case, we can still derive meaningful cluster
assignments. Existing works only consider complete anomalies. In this paper, we present a novel probabilistic generative model (PAICAN) that explicitly models partial anomalies
by generalizing ideas of Degree Corrected Stochastic Block
Models and Bernoulli Mixture Models. We provide a highly
scalable variational inference approach with runtime complexity linear in the number of edges. The robustness of our
model w.r.t. anomalies is demonstrated by our experimental
study, outperforming state-of-the-art competitors.


- **14.A Poisson Gamma Probabilistic Model for Latent Node-group Memberships in Dynamic Networks (AAAI 2018)**
  - Sikun Yang and Heinz Koeppl
  - [[Paper]](https://arxiv.org/pdf/1805.11054.pdf)
  - [[C Reference]](https://github.com/stephenyang/dynamic-Edge-Partition-Models)

We present a probabilistic model for learning from dynamic
relational data, wherein the observed interactions among networked nodes are modeled via the Bernoulli Poisson link
function, and the underlying network structure are characterized by nonnegative latent node-group memberships, which
are assumed to be gamma distributed. The latent memberships evolve according to Markov processes. The optimal
number of latent groups can be determined by data itself.
The computational complexity of our method scales with the
number of non-zero links, which makes it scalable to large
sparse dynamic relational data. We present batch and online
Gibbs sampling algorithms to perform model inference. Finally, we demonstrate the model’s performance on both synthetic and real-world datasets compared to state-of-the-art
methods

- **15.Sentiment-driven Community Profiling and Detection on Social Media (ACM HSM 2018)**
  - Amin Salehi, Mert Ozer, and Hasan Davulcu
  - [[Paper]](https://arxiv.org/pdf/1810.06917v1.pdf)
  - [[Matlab Reference]](https://github.com/amin-salehi/GSNMF)
  
- **16.TNE: A Latent Model for Representation Learning on Networks (Arxiv 2018)**
  - Abdulkadir Çelikkanat and Fragkiskos D. Malliaros 
  - [[Paper]](https://arxiv.org/pdf/1611.06645.pdf)
  - [[Python Reference]](https://github.com/abdcelikkanat/TNE)

Network representation learning (NRL) methods aim to map each
vertex into a low dimensional space by preserving the local and
global structure of a given network, and in recent years they have
received a significant attention thanks to their success in several
challenging problems. Although various approaches have been
proposed to compute node embeddings, many successful methods
benefit from random walks in order to transform a given network
into a collection of sequences of nodes and then they target to
learn the representation of nodes by predicting the context of
each vertex within the sequence. In this paper, we introduce a
general framework to enhance the embeddings of nodes acquired
by means of the random walk-based approaches. Similar to the
notion of topical word embeddings in NLP, the proposed method
assigns each vertex to a topic with the favor of various statistical
models and community detection methods, and then generates the
enhanced community representations. We evaluate our method on
two downstream tasks: node classification and link prediction. The
experimental results demonstrate that the incorporation of vertex
and topic embeddings outperform widely-known baseline NRL
methods.

- **17.A Unified Framework for Community Detection and Network Representation Learning**
  - [[Paper]](https://arxiv.org/pdf/1611.06645.pdf)

Abstract—Network representation learning (NRL) aims to learn low-dimensional vectors for vertices in a network. Most existing NRL
methods focus on learning representations from local context of vertices (such as their neighbors). Nevertheless, vertices in many
complex networks also exhibit significant global patterns widely known as communities. It’s intuitive that vertices in the same
community tend to connect densely and share common attributes. These patterns are expected to improve NRL and benefit relevant
evaluation tasks, such as link prediction and vertex classification. Inspired by the analogy between network representation learning and
text modeling, we propose a unified NRL framework by introducing community information of vertices, named as Community-enhanced
Network Representation Learning (CNRL). CNRL simultaneously detects community distribution of each vertex and learns
embeddings of both vertices and communities. Moreover, the proposed community enhancement mechanism can be applied to various
existing NRL models. In experiments, we evaluate our model on vertex classification, link prediction, and community detection using
several real-world datasets. The results demonstrate that CNRL significantly and consistently outperforms other state-of-the-art
methods while verifying our assumptions on the correlations between vertices and communities.

- **18.Non-Linear Attributed Graph Clustering by Symmetric NMF with PU Learning (Arxiv 2018)**
  - Seiji Maekawa, Koh Takeuch, Makoto Onizuka
  - [[Paper]](https://arxiv.org/abs/1810.00946)
  - [[Python Reference]](https://github.com/seijimaekawa/NAGC)

We consider the clustering problem of attributed graphs. Our challenge is how we can design an effective and efficient clustering method that precisely captures the hidden relationship between the topology and the attributes in real-world graphs. We propose Non-linear Attributed Graph Clustering by Symmetric Non-negative Matrix Factorization with Positive Unlabeled Learning. The features of our method are three holds. 1) it learns a non-linear projection function between the different cluster assignments of the topology and the attributes of graphs so as to capture the complicated relationship between the topology and the attributes in real-world graphs, 2) it leverages the positive unlabeled learning to take the effect of partially observed positive edges into the cluster assignment, and 3) it achieves efficient computational complexity, O((n2+mn)kt), where n is the vertex size, m is the attribute size, k is the number of clusters, and t is the number of iterations for learning the cluster assignment. We conducted experiments extensively for various clustering methods with various real datasets to validate that our method outperforms the former clustering methods regarding the clustering quality.

- **19.A Nonnegative Matrix Factorization Approach for Multiple Local Community Detection (ASONAM 2018)**
  - Dany Kamuhanda and Kun He
  - [[Paper]](https://www.researchgate.net/publication/326208243_A_Nonnegative_Matrix_Factorization_Approach_for_Multiple_Local_Community_Detection)
  - [[Python Reference]](https://github.com/danison2/MLC-code)

Existing works on local community detection in social networks focus on finding one single community a few seed members are most likely to be in. In this work, we address a much harder problem of multiple local community detection and propose a Nonnegative Matrix Factorization algorithm for finding multiple local communities for a single seed chosen randomly in multiple ground truth communities. The number of detected communities for the seed is determined automatically by the algorithm. We first apply a Breadth-First Search to sample the input graph up to several levels depending on the network density. We then use Nonnegative Matrix Factorization on the adjacency matrix of the sampled subgraph to estimate the number of communities, and then cluster the nodes of the subgraph into communities. Our proposed method differs from the existing NMF-based community detection methods as it does not use "argmax" function to assign nodes to communities. Our method has been evaluated on real-world networks and shows good accuracy as evaluated by the F1 score when comparing with the state-of-the-art local community detection algorithm.

- **20.Community Preserving Network Embedding (AAAI 17)**
  - Xiao Wang, Peng Cui, Jing Wang, Jain Pei, WenWu Zhu, Shiqiang Yang
  - [[Paper]](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14589/13763)
  - [[Python Reference]](https://github.com/benedekrozemberczki/M-NMF)
  - [[Matlab Reference]](https://github.com/AnryYang/M-NMF)


Network embedding, aiming to learn the low-dimensional
representations of nodes in networks, is of paramount importance in many real applications. One basic requirement of
network embedding is to preserve the structure and inherent
properties of the networks. While previous network embedding methods primarily preserve the microscopic structure,
such as the first- and second-order proximities of nodes, the
mesoscopic community structure, which is one of the most
prominent feature of networks, is largely ignored. In this paper, we propose a novel Modularized Nonnegative Matrix
Factorization (M-NMF) model to incorporate the community structure into network embedding. We exploit the consensus relationship between the representations of nodes and
community structure, and then jointly optimize NMF based
representation learning model and modularity based community detection model in a unified framework, which enables
the learned representations of nodes to preserve both of the
microscopic and community structures. We also provide efficient updating rules to infer the parameters of our model, together with the correctness and convergence guarantees. Extensive experimental results on a variety of real-world networks show the superior performance of the proposed method
over the state-of-the-arts.

- **21.A Non-negative Symmetric Encoder-Decoder Approach for Community Detection (CIKM 17)**
  - Bing-Jie Sun, Huawei Shen, Jinhua Gao, Wentao Ouyang, Xueqi Cheng
  - [[Paper]](http://www.bigdatalab.ac.cn/~shenhuawei/publications/2017/cikm-sun.pdf)
  - [[Python Reference]](https://github.com/benedekrozemberczki/karateclub)

Community detection or graph clustering is crucial to understanding the structure of complex networks and extracting relevant knowledge from networked data. Latent factor
model, e.g., non-negative matrix factorization and mixed
membership block model, is one of the most successful methods for community detection. Latent factor models for community detection aim to find a distributed and generally
low-dimensional representation, or coding, that captures the
structural regularity of network and reflects the community
membership of nodes. Existing latent factor models are mainly
based on reconstructing a network from the representation
of its nodes, namely network decoder, while constraining the
representation to have certain desirable properties. These
methods, however, lack an encoder that transforms nodes
into their representation. Consequently, they fail to give a
clear explanation about the meaning of a community and
suffer from undesired computational problems. In this paper, we propose a non-negative symmetric encoder-decoder
approach for community detection. By explicitly integrating
a decoder and an encoder into a unified loss function, the
proposed approach achieves better performance over stateof-the-art latent factor models for community detection task.
Moreover, different from existing methods that explicitly impose the sparsity constraint on the representation of nodes,
the proposed approach implicitly achieves the sparsity of
node representation through its symmetric and non-negative
properties, making the optimization much easier than competing methods based on sparse matrix factorization


- **22.Self-weighted Multiview Clustering with Multiple Graphs (IJCAI 17)**
  - Feiping Nie, Jing Li, and Xuelong Li
  - [[Paper]](https://www.ijcai.org/proceedings/2017/0357.pdf)
  - [[Matlab Reference]](https://github.com/kylejingli/SwMC-IJCAI17)

In multiview learning, it is essential to assign a reasonable weight to each view according to the view
importance. Thus, for multiview clustering task,
a wise and elegant method should achieve clustering multiview data while learning the view weights. In this paper, we propose to explore a Laplacian
rank constrained graph, which can be approximately as the centroid of the built graph for each view
with different confidences. We start our work with
a natural thought that the weights can be learned
by introducing a hyperparameter. By analyzing the
weakness of this way, we further propose a new
multiview clustering method which is totally selfweighted. More importantly, once the target graph
is obtained in our models, we can directly assign
the cluster label to each data point and do not need
any postprocessing such as K-means in standard
spectral clustering. Evaluations on two synthetic datasets indicate the effectiveness of our methods. Compared with several representative graphbased multiview clustering approaches on four realworld datasets, the proposed methods achieve the
better performances and our new clustering method
is more practical to use.

- **23.Semi-supervised Clustering in Attributed Heterogeneous Information Networks (WWW 17)**
  - Xiang Li, Yao Wu, Martin Ester, Ben Kao, Xin Wang, and Yudian Zheng
  - [[Paper]](https://dl.acm.org/citation.cfm?id=3052576)
  - [[Python Reference]](https://github.com/wedoso/SCHAIN-NL)

A heterogeneous information network (HIN) is one whose nodes model objects of different types and whose links model objects' relationships. In many applications, such as social networks and RDF-based knowledge bases, information can be modeled as HINs. To enrich its information content, objects (as represented by nodes) in an HIN are typically associated with additional attributes. We call such an HIN an Attributed HIN or AHIN. We study the problem of clustering objects in an AHIN, taking into account objects' similarities with respect to both object attribute values and their structural connectedness in the network. We show how supervision signal, expressed in the form of a must-link set and a cannot-link set, can be leveraged to improve clustering results. We put forward the SCHAIN algorithm to solve the clustering problem. We conduct extensive experiments comparing SCHAIN with other state-of-the-art clustering algorithms and show that SCHAIN outperforms the others in clustering quality.

- **24.Learning Community Embedding with Community Detection and Node Embedding on Graph (CIKM 2017)**
  - Sandro Cavallari, Vincent W. Zheng, Hongyun Cai, Kevin Chen-Chuan Chang, and Erik Cambria
  - [[Paper]](http://sentic.net/community-embedding.pdf)
  - [[Python Reference]](https://github.com/andompesta/ComE)

In this paper, we study an important yet largely under-explored
setting of graph embedding, i.e., embedding communities instead
of each individual nodes. We find that community embedding is
not only useful for community-level applications such as graph
visualization, but also beneficial to both community detection and
node classification. To learn such embedding, our insight hinges
upon a closed loop among community embedding, community detection and node embedding. On the one hand, node embedding
can help improve community detection, which outputs good communities for fitting better community embedding. On the other
hand, community embedding can be used to optimize the node embedding by introducing a community-aware high-order proximity.
Guided by this insight, we propose a novel community embedding
framework that jointly solves the three tasks together. We evaluate
such a framework on multiple real-world datasets, and show that
it improves graph visualization and outperforms state-of-the-art
baselines in various application tasks, e.g., community detection
and node classification.

- **25.Cross-Validation Estimate of the Number of Clusters in a Network (Scientific Report 2017)**
  - Matsuro Kawamoto and Yoshiyuki Kabashima
  - [[Paper]](https://arxiv.org/abs/1605.07915)
  - [[Julia Reference]](https://github.com/tatsuro-kawamoto/graphBIX)

Network science investigates methodologies that summarise relational data to obtain better interpretability. Identifying modular structures is a fundamental task, and assessment of the coarse-grain level is its crucial step. Here, we propose principled, scalable, and widely applicable assessment criteria to determine the number of clusters in modular networks based on the leave-one-out cross-validation estimate of the edge prediction error.

- **26.Comparative Analysis on the Selection of Number of Clusters in Community Detection (ArXiv 2017)**
  - Matsuro Kawamoto and Yoshiyuki Kabashima
  - [[Paper]](https://arxiv.org/abs/1606.07668)
  - [[Julia Reference]](https://github.com/tatsuro-kawamoto/graphBIX)

We conduct a comparative analysis on various estimates of the number of clusters in community detection. An exhaustive comparison requires testing of all possible combinations of frameworks, algorithms, and assessment criteria. In this paper we focus on the framework based on a stochastic block model, and investigate the performance of greedy algorithms, statistical inference, and spectral methods. For the assessment criteria, we consider modularity, map equation, Bethe free energy, prediction errors, and isolated eigenvalues. From the analysis, the tendency of overfit and underfit that the assessment criteria and algorithms have, becomes apparent. In addition, we propose that the alluvial diagram is a suitable tool to visualize statistical inference results and can be useful to determine the number of clusters.

- **27.Subspace Based Network Community Detection Using Sparse Linear Coding (TKDE 2016)**
  - Arif Mahmood and Michael Small 
  - [[Paper]](https://ieeexplore.ieee.org/document/7312985)
  - [[Python Reference]](https://github.com/DamonLiuTHU/Subspace-Based-Network-Community-Detection-Using-Sparse-Linear-Coding)

 Information mining from networks by identifying communities is an important problem across a number of research fields including social science, biology, physics, and medicine. Most existing community detection algorithms are graph theoretic and lack the ability to detect accurate community boundaries if the ratio of intra-community to inter-community links is low. Also, algorithms based on modularity maximization may fail to resolve communities smaller than a specific size if the community size varies significantly. In this paper, we present a fundamentally different community detection algorithm based on the fact that each network community spans a different subspace in the geodesic space. Therefore, each node can only be efficiently represented as a linear combination of nodes spanning the same subspace. To make the process of community detection more robust, we use sparse linear coding with l 1 norm constraint. In order to find a community label for each node, sparse spectral clustering algorithm is used. The proposed community detection technique is compared with more than 10 state of the art methods on two benchmark networks (with known clusters) using normalized mutual information criterion. Our proposed algorithm outperformed existing algorithms with a significant margin on both benchmark networks. The proposed algorithm has also shown excellent performance on three real-world networks.

- **28.Joint Community and Structural Hole Spanner Detection via Harmonic Modularity (KDD 2016)**
  - Lifang He, Chun-Ta Lu, Jiaqi Mu, Jianping Cao, Linlin Shen, and Philip S Yu
  - [[Paper]](https://www.kdd.org/kdd2016/papers/files/rfp1184-heA.pdf)
  - [[Python Reference]](https://github.com/LifangHe/KDD16_HAM)

Detecting communities (or modular structures) and structural hole spanners, the nodes bridging different communities in a network, are two essential tasks in the realm of
network analytics. Due to the topological nature of communities and structural hole spanners, these two tasks are
naturally tangled with each other, while there has been little synergy between them. In this paper, we propose a novel
harmonic modularity method to tackle both tasks simultaneously. Specifically, we apply a harmonic function to measure the smoothness of community structure and to obtain
the community indicator. We then investigate the sparsity
level of the interactions between communities, with particular emphasis on the nodes connecting to multiple communities, to discriminate the indicator of SH spanners and assist
the community guidance. Extensive experiments on realworld networks demonstrate that our proposed method outperforms several state-of-the-art methods in the community
detection task and also in the SH spanner identification task
(even the methods that require the supervised community
information). Furthermore, by removing the SH spanners
spotted by our method, we show that the quality of other
community detection methods can be further improved.

- **29.Community Detection via Fused Loadings Principal Component Analysis (2016)**
  - Richard Samworth, Yang Feng, and Yi Yu
  - [[R Reference]](https://github.com/cran/FusedPCA)

- **30.Feature Extraction via Multi-view Non-negative Matrix Factorization with Local Graph Regularization (IEEE ICIP 2015)**
  - Zhenfan Wang, Xiangwei Kong, Hiayan Fu, Ming Li, and Yujia Zhang
  - [[Paper]](https://ieeexplore.ieee.org/document/7351455)
  - [[Matlab Reference]](https://github.com/DUT-DIPLab/Graph-Multi-NMF-Feature-Clustering)


Feature extraction is a crucial and difficult issue in pattern recognition tasks with the high-dimensional and multiple features. To extract the latent structure of multiple features without label information, multi-view learning algorithms have been developed. In this paper, motivated by manifold learning and multi-view Non-negative Matrix Factorization (NM-F), we introduce a novel feature extraction method via multi-view NMF with local graph regularization, where the inner-view relatedness between data is taken into consideration. We propose the matrix factorization objective function by constructing a nearest neighbor graph to integrate local geometrical information of each view and apply two iterative updating rules to effectively solve the optimization problem. In the experiment, we use the extracted feature to cluster several realistic datasets. The experimental results demonstrate the effectiveness of our proposed feature extraction approach.

- **31.A Uniﬁed Semi-Supervised Community Detection Framework Using Latent Space Graph Regularization (IEEE TOC 2015)**
  - Liang Yang, Xiaochun Cao, Di Jin, Xiao Wang, and Dan Meng
  - [[Paper]](http://yangliang.github.io/pdf/06985550.pdf)
  - [[Matlab Reference]](http://yangliang.github.io/code/LSGR.rar)


Community structure is one of the most important
properties of complex networks and is a foundational concept in
exploring and understanding networks. In real world, topology
information alone is often inadequate to accurately find community structure due to its sparsity and noises. However, potential
useful prior information can be obtained from domain knowledge in many applications. Thus, how to improve the community
detection performance by combining network topology with prior
information becomes an interesting and challenging problem.
Previous efforts on utilizing such priors are either dedicated
or insufficient. In this paper, we firstly present a unified interpretation to a group of existing community detection methods.
And then based on this interpretation, we propose a unified
semi-supervised framework to integrate network topology with
prior information for community detection. If the prior information indicates that some nodes belong to the same community, we
encode it by adding a graph regularization term to penalize the
latent space dissimilarity of these nodes. This framework can be
applied to many widely-used matrix-based community detection
methods satisfying our interpretation, such as nonnegative matrix
factorization, spectral clustering, and their variants. Extensive
experiments on both synthetic and real networks show that the
proposed framework significantly improves the accuracy of community detection, especially on networks with unclear structures.

- **32.Community Detection via Measure Space Embedding (NIPS 2015)**
  - Yulong Pei, Nilanjan Chakraborty, and Katia Sycara
  - [[Paper]](https://papers.nips.cc/paper/5808-community-detection-via-measure-space-embedding.pdf)
  - [[Python Reference]](https://github.com/komarkdev/der_graph_clustering)

![image](https://raw.githubusercontent.com/JimWongM/ImageHost/master/img/20200408091919.png)

- **33.Nonnegative Matrix Tri-Factorization with Graph Regularization for Community Detection in Social Networks (IJCAI 2015)**
  - Mark Kozdoba and Shie Mannor
  - [[Paper]](https://www.ijcai.org/Proceedings/15/Papers/295.pdf)
  - [[Python Reference]](https://github.com/yunhenk/NMTF)

Community detection on social media is a classic
and challenging task. In this paper, we study the
problem of detecting communities by combining
social relations and user generated content in social networks. We propose a nonnegative matrix
tri-factorization (NMTF) based clustering framework with three types of graph regularization. The
NMTF based clustering framework can combine
the relations and content seamlessly and the graph
regularization can capture user similarity, message
similarity and user interaction explicitly. In order to design regularization components, we further exploit user similarity and message similarity
in social networks. A unified optimization problem is proposed by integrating the NMTF framework and the graph regularization. Then we derive
an iterative learning algorithm for this optimization
problem. Extensive experiments are conducted on
three real-world data sets and the experimental results demonstrate the effectiveness of the proposed
method.

- **34.Community Detection for Clustered Attributed Graphs via a Variational EM Algorithm (Big Data 2014)**
  - Xiangyong Cao, Xiangyu Chang, and Zongben Xu
  - [[Paper]](https://dl.acm.org/citation.cfm?id=2644179)
  - [[Matlab Reference]](https://github.com/xiangyongcao/Variational-EM-for-Community-Detection)
  
- **35.Improved Graph Clustering (Transactions on Information Network Theory 2014)**
  - Yudong Chen, Sujay Sanghavi, Huan Xu 
  - [[Paper]](https://ieeexplore.ieee.org/document/6873307)
  - [[Matlab Reference]](https://github.com/sara-karami/improved_graph_clustering)

Graph clustering involves the task of dividing nodes into clusters, so that the edge density is higher within clusters as opposed to across clusters. A natural, classic, and popular statistical setting for evaluating solutions to this problem is the stochastic block model, also referred to as the planted partition model. In this paper, we present a new algorithm-a convexified version of maximum likelihood-for graph clustering. We show that, in the classic stochastic block model setting, it outperforms existing methods by polynomial factors when the cluster size is allowed to have general scalings. In fact, it is within logarithmic factors of known lower bounds for spectral methods, and there is evidence suggesting that no polynomial time algorithm would do significantly better. We then show that this guarantee carries over to a more general extension of the stochastic block model. Our method can handle the settings of semirandom graphs, heterogeneous degree distributions, unequal cluster sizes, unaffiliated nodes, partially observed graphs, planted clique/coloring, and so on. In particular, our results provide the best exact recovery guarantees to date for the planted partition, planted k-disjoint-cliques and planted noisy coloring models with general cluster sizes; in other settings, we match the best existing results up to logarithmic factors.

- **36.Overlapping Community Detection at Scale: a Nonnegative Matrix Factorization Approach (WSDM 2013)**
  - Jaewon Yang and Jure Leskovec
  - [[Paper]](http://i.stanford.edu/~crucis/pubs/paper-nmfagm.pdf)
  - [[C++ Reference]](https://github.com/snap-stanford/snap/tree/master/examples/bigclam)
  - [[Java Spark Reference]](https://github.com/thangdnsf/BigCLAM-ApacheSpark)
  - [[Python Reference]](https://github.com/benedekrozemberczki/karateclub)
  - [[Python Reference]](https://github.com/RobRomijnders/bigclam)
  - [[Python Reference]](https://github.com/jeremyzhangsq/map-reduce-bigclam)  

Network communities represent basic structures for understanding
the organization of real-world networks. A community (also referred to as a module or a cluster) is typically thought of as a group
of nodes with more connections amongst its members than between
its members and the remainder of the network. Communities in
networks also overlap as nodes belong to multiple clusters at once.
Due to the difficulties in evaluating the detected communities and
the lack of scalable algorithms, the task of overlapping community
detection in large networks largely remains an open problem.
In this paper we present BIGCLAM (Cluster Affiliation Model
for Big Networks), an overlapping community detection method
that scales to large networks of millions of nodes and edges. We
build on a novel observation that overlaps between communities
are densely connected. This is in sharp contrast with present community detection methods which implicitly assume that overlaps
between communities are sparsely connected and thus cannot properly extract overlapping communities in networks. In this paper,
we develop a model-based community detection algorithm that can
detect densely overlapping, hierarchically nested as well as nonoverlapping communities in massive networks. We evaluate our algorithm on 6 large social, collaboration and information networks
with ground-truth community information. Experiments show state
of the art performance both in terms of the quality of detected communities as well as in speed and scalability of our algorithm.


- **37.On the Statistical Detection of Clusters in Undirected Networks (Computation Statistics and Data Analysis 2013)**
  - Marcus B. Perry, Gregory V. Michaelson, M. Allan Ballard
  - [[Paper]](https://dl.acm.org/citation.cfm?id=2750189)
  - [[C++ Reference]](https://github.com/alanballard/Likelihood-Based-Directed-Network-Clustering)

The goal of network clustering algorithms is to assign each node in a network to one of several mutually exclusive groups based upon the observed edge set. Of the network clustering algorithms widely available, most make the effort to maximize the modularity metric. Although modularity is an intuitive and effective means to cluster networks, it provides no direct basis for quantifying the statistical significance of the detected clusters. In this paper, we consider undirected networks and propose a new objective function to maximize over the space of possible group membership assignments. This new objective function lends naturally to the use of information criterion (e.g., Akaike or Bayesian) for determining the "best" number of groups, as well as to the development of a likelihood ratio test for determining if the clusters detected provide significant new information. The proposed method is demonstrated using two real-world networks. Additionally, using Monte Carlo simulation, we compare the performances of the proposed clustering framework relative to that achieved by maximizing the modularity objective when applied to LFR benchmark graphs.

- **38.Symmetric Nonnegative Matrix Factorization for Graph Clustering (SDM 2012)**
  - Da Kuang, Chris Ding, and Haesun Park
  - [[Paper]](https://www.cc.gatech.edu/~hpark/papers/DaDingParkSDM12.pdf)
  - [[Matlab Reference]](https://github.com/dakuang/symnmf)

Nonnegative matrix factorization (NMF) provides a lower
rank approximation of a nonnegative matrix, and has been
successfully used as a clustering method. In this paper, we
offer some conceptual understanding for the capabilities and
shortcomings of NMF as a clustering method. Then, we
propose Symmetric NMF (SymNMF) as a general framework for graph clustering, which inherits the advantages of
NMF by enforcing nonnegativity on the clustering assignment matrix. Unlike NMF, however, SymNMF is based
on a similarity measure between data points, and factorizes a symmetric matrix containing pairwise similarity values (not necessarily nonnegative). We compare SymNMF
with the widely-used spectral clustering methods, and give
an intuitive explanation of why SymNMF captures the cluster structure embedded in the graph representation more
naturally. In addition, we develop a Newton-like algorithm
that exploits second-order information efficiently, so as to
show the feasibility of SymNMF as a practical framework
for graph clustering. Our experiments on artificial graph
data, text data, and image data demonstrate the substantially enhanced clustering quality of SymNMF over spectral
clustering and NMF. Therefore, SymNMF is able to achieve
better clustering results on both linear and nonlinear manifolds, and serves as a potential basis for many extensions
and applications.

- **39.A Model-based Approach to Attributed Graph Clustering (SIGMOID 2012)**
  - Zhiqiang Xu, Yiping Ke, Yi Wang, Hong Cheng, and James Cheng
  - [[Paper]](http://www-std1.se.cuhk.edu.hk/~hcheng/paper/BAGC_sigmod12.pdf)
  - [[Matlab Reference]](https://github.com/zhiqiangxu2001/BAGC)

Graph clustering, also known as community detection, is a longstanding problem in data mining. However, with the proliferation of rich attribute information available for objects in real-world
graphs, how to leverage structural and attribute information for
clustering attributed graphs becomes a new challenge. Most existing works take a distance-based approach. They proposed various distance measures to combine structural and attribute information. In this paper, we consider an alternative view and propose a
model-based approach to attributed graph clustering. We develop
a Bayesian probabilistic model for attributed graphs. The model
provides a principled and natural framework for capturing both
structural and attribute aspects of a graph, while avoiding the artificial design of a distance measure. Clustering with the proposed
model can be transformed into a probabilistic inference problem,
for which we devise an efficient variational algorithm. Experimental results on large real-world datasets demonstrate that our method
significantly outperforms the state-of-art distance-based attributed
graph clustering method

- **40.Overlapping Community Detection Using Bayesian Non-negative Matrix Factorization (Physical Review E 2011)**
  - Ionnis Psorakis, Stephen Roberts, Mark Ebden, and Ben Sheldon
  - [[Paper]](http://www.orchid.ac.uk/eprints/38/1/PRE_NMF.pdf)
  - [[Matlab Reference]](https://github.com/ipsorakis/commDetNMF)

Identifying overlapping communities in networks is a challenging task. In this work we present a novel
approach to community detection that utilizes a Bayesian nonnegative matrix factorization (NMF) model to
extract overlapping modules from a network. The scheme has the advantage of soft-partitioning solutions,
assignment of node participation scores to modules and an intuitive foundation. We present the performance of
the method against a variety of benchmark problems and compare and contrast it to several other algorithms for
community detection