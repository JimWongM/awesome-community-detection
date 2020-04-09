## Label Propagation, Percolation and Random Walks

- **1.Nonlinear Diffusion for Community Detection and Semi-Supervised Learning (WWW 2019)**
  - Rania Ibrahim, David F. Gleich
  - [[Paper]](https://dl.acm.org/doi/10.1145/3308558.3313483)
  - [[Python Reference]](https://github.com/RaniaSalama/Nonlinear_Diffusion)

Diffusions, such as the heat kernel diffusion and the PageRank vector, and their relatives are widely used graph mining primitives that have been successful in a variety of contexts including community detection and semi-supervised learning. The majority of existing methods and methodology involves linear diffusions, which then yield simple algorithms involving repeated matrix-vector operations. Recent work, however, has shown that sophisticated and complicated techniques based on network embeddings and neural networks can give empirical results superior to those based on linear diffusions. In this paper, we illustrate a class of nonlinear graph diffusions that are competitive with state of the art embedding techniques and outperform classic diffusions. Our new methods enjoy much of the simplicity underlying classic diffusion methods as well. Formally, they are based on nonlinear dynamical systems that can be realized with an implementation akin to applying a nonlinear function after each matrix-vector product in a classic diffusion. This framework also enables us to easily integrate results from multiple data representations in a principled fashion. Furthermore, we have some theoretical relationships that suggest choices of the nonlinear term. We demonstrate the benefits of these techniques on a variety of synthetic and real-world data.

- **2.Community Detection in Bipartite Networks by Multi Label Propagation Algorithm (JSAI 2019)**
  - Hibiki Taguchi, Tsuyoshi Murata
  - [[Paper]](https://confit.atlas.jp/guide/event/jsai2019/subject/4B2-J-3-02/detail)
  - [[Python Reference]](https://github.com/hbkt/BiMLPA)

Community detection is an important topic in complex networks.
A bipartite network is a special type of network, whose nodes can be divided into two disjoint sets and each edge connects between different types of nodes. In bipartite networks, there are two types of community definition, one-to-one correspondence and many-to-many correspondence between communities. The latter is better to represent realistic community structure in bipartite networks. However, few method can extract this type of structure.
In this paper, we propose BiMLPA, based on multi label propagation algorithm, to detect many-to-many correspondence between communities in bipartite networks. Experimental results on real-world networks show that BiMLPA is effective and stable for detecting communities.

- **3.Constrained Local Graph Clustering by Colored Random Walk (WWW 2019)**
  - Yaowei Yan, Yuchen Bian, Dongsheng Luo, Dongwon Lee and Xiang Zhang
  - [[Paper]](http://pike.psu.edu/publications/www19.pdf)
  - [[Matlab Reference]](https://github.com/yanyaw/colored-random-walk)

Detecting local graph clusters is an important problem in big graph
analysis. Given seed nodes in a graph, local clustering aims at finding subgraphs around the seed nodes, which consist of nodes highly
relevant to the seed nodes. However, existing local clustering methods either allow only a single seed node, or assume all seed nodes
are from the same cluster, which is not true in many real applications. Moreover, the assumption that all seed nodes are in a single
cluster fails to use the crucial information of relations between seed
nodes. In this paper, we propose a method to take advantage of such
relationship. With prior knowledge of the community membership
of the seed nodes, the method labels seed nodes in the same (different) community by the same (different) color. To further use this
information, we introduce a color-based random walk mechanism,
where colors are propagated from the seed nodes to every node in
the graph. By the interaction of identical and distinct colors, we can
enclose the supervision of seed nodes into the random walk process.
We also propose a heuristic strategy to speed up the algorithm by
more than 2 orders of magnitude. Experimental evaluations reveal
that our clustering method outperforms state-of-the-art approaches
by a large margin.

- **4.Dynamic Graph-Based Label Propagation for Density Peaks Clustering (Expert Systems 2019)**
  - Seyed Amjad Seyedi, Abdulrahman Lotfi, Parham Moradi and Nooruldeen Nasih Qader
  - [[Paper]](https://www.sciencedirect.com/science/article/pii/S0957417418304998?via%3Dihub)
  - [[Matlab Reference]](https://github.com/amjadseyedi/DPC-DLP)

Clustering is a major approach in data mining and machine learning and has been successful in many real-world applications. Density peaks clustering (DPC) is a recently published method that uses an intuitive to cluster data objects efficiently and effectively. However, DPC and most of its improvements suffer from some shortcomings to be addressed. For instance, this method only considers the global structure of data which leading to missing many clusters. The cut-off distance affects the local density values and is calculated in different ways depending on the size of the datasets, which can influence the quality of clustering. Then, the original label assignment can cause a “chain reaction”, whereby if a wrong label is assigned to a data point, and then there may be many more wrong labels subsequently assigned to the other points. In this paper, a density peaks clustering method called DPC-DLP is proposed. The proposed method employs the idea of k-nearest neighbors to compute the global cut-off parameter and the local density of each point. Moreover, the proposed method uses a graph-based label propagation to assign labels to remaining points and form final clusters. The proposed label propagation can effectively assign true labels to those of data instances which located in border and overlapped regions. The proposed method can be applied to some applications. To make the method practical for image clustering, the local structure is used to achieve low-dimensional space. In addition, proposed method considers label space correlation, to be effective in the gene expression problems. Several experiments are performed to evaluate the performance of the proposed method on both synthetic and real-world datasets. The results demonstrate that in most cases, the proposed method outperformed some state-of-the-art methods.

- **5.Community Detection by Information Flow Simulation (ArXiv 2018)**
  - Rajagopal Venkatesaramani and Yevgeniy Vorobeychik 
  - [[Paper]](https://arxiv.org/abs/1805.04920)
  - [[Python Reference]](https://github.com/rajagopalvenkat/Community_Detection-Flow_Simulation)

Community detection remains an important problem in data mining, owing to the lack of scalable algorithms that exploit all aspects of available data - namely the directionality of flow of information and the dynamics thereof. Most existing methods use measures of connectedness in the graphical structure. In this paper, we present a fast, scalable algorithm to detect communities in directed, weighted graph representations of social networks by simulating flow of information through them. By design, our algorithm naturally handles undirected or unweighted networks as well. Our algorithm runs in O(|E|) time, which is better than most existing work and uses O(|E|) space and hence scales easily to very large datasets. Finally, we show that our algorithm outperforms the state-of-the-art Markov Clustering Algorithm (MCL) in both accuracy and scalability on ground truth data (in a number of cases, we can find communities in graphs too large for MCL).

- **6.Multiple Local Community Detection (ACM SIGMETRICS 2017)**
  - Alexandre Hollocou, Thomas Bonald, and Marc Lelarge
  - [[Paper]](https://hal.archives-ouvertes.fr/hal-01625444)
  - [[Python Reference]](https://github.com/ahollocou/multicom)

 Community detection is a classical problem in the field of graph mining. We are interested in local community detection where the objective is the recover the communities containing some given set of nodes, called the seed set. While existing approaches typically recover only one community around the seed set, most nodes belong to multiple communities in practice. In this paper, we introduce a new algorithm for detecting multiple local communities, possibly overlapping, by expanding the initial seed set. The new nodes are selected by some local clustering of the graph embedded in a vector space of low dimension. We validate our approach on real graphs, and show that it provides more information than existing algorithms to recover the complex graph structure that appears locally.

- **7.Krylov Subspace Approximation for Local Community Detection in Large Networks (ArXiv 2017)**
  - Kun He, Pan Shi, David Bindel, and John E. Hopcroft
  - [[Paper]](https://arxiv.org/pdf/1712.04823.pdf)
  - [[Matlab Reference]](https://github.com/PanShi2016/LOSP_Plus)

Community detection is an important information mining task to uncover modular structures in large networks.
For increasingly common large network data sets, global community detection is prohibitively expensive, and
attention has shifted to methods that mine local communities, i.e. identifying all latent members of a particular
community from a few labeled seed members. To address such semi-supervised mining task, we systematically
develop a local spectral subspace-based community detection method, called LOSP. We define a family of
local spectral subspaces based on Krylov subspaces, and seek a sparse indicator for the target community
via an ℓ1 norm minimization over the Krylov subspace. Variants of LOSP depend on type of random walks
with different diffusion speeds, type of random walks, dimension of the local spectral subspace and step of
diffusions. The effectiveness of the proposed LOSP approach is theoretically analyzed based on Rayleigh
quotients, and it is experimentally verified on a wide variety of real-world networks across social, production
and biological domains, as well as on an extensive set of synthetic LFR benchmark datasets

- **8.Many Heads are Better than One: Local Community Detection by the Multi-Walker Chain (ICDM 2017)**
  - Yuchen Bian, Jingchao Ni, Wei Cheng, and Zhang Xiang
  - [[Paper]](https://ieeexplore.ieee.org/document/8215474)
  - [[C++ Reference]](https://github.com/flyingdoog/MWC)

Local community detection (or local clustering) is of fundamental importance in large network analysis. Random walk based methods have been routinely used in this task. Most existing random walk methods are based on the single-walker model. However, without any guidance, a single-walker may not be adequate to effectively capture the local cluster. In this paper, we study a multi-walker chain (MWC) model, which allows multiple walkers to explore the network. Each walker is influenced (or pulled back) by all other walkers when deciding the next steps. This helps the walkers to stay as a group and within the cluster. We introduce two measures based on the mean and standard deviation of the visiting probabilities of the walkers. These measures not only can accurately identify the local cluster, but also help detect the cluster center and boundary, which cannot be achieved by the existing single-walker methods. We provide rigorous theoretical foundation for MWC, and devise efficient algorithms to compute it. Extensive experimental results on a variety of real-world networks demonstrate that MWC outperforms the state-of-the-art local community detection methods by a large margin.

- **9.Improving PageRank for Local Community Detection (ArXiv 2016)**
  - Alexandre Hollocou, Thomas Bonald, and Marc Lelarge
  - [[Paper]](https://arxiv.org/abs/1610.08722)
  - [[C Reference]](https://github.com/ahollocou/walkscan)
  - [[Python Reference]](https://github.com/ahollocou/walkscan)

Community detection is a classical problem in the field of graph mining. While most algorithms work on the entire graph, it is often interesting in practice to recover only the community containing some given set of seed nodes. In this paper, we propose a novel approach to this problem, using some low-dimensional embedding of the graph based on random walks starting from the seed nodes. From this embedding, we propose some simple yet efficient versions of the PageRank algorithm as well as a novel algorithm, called WalkSCAN, that is able to detect multiple communities, possibly overlapping. We provide insights into the performance of these algorithms through the theoretical analysis of a toy network and show that WalkSCAN outperforms existing algorithms on real networks.


- **10.Limited Random Walk Algorithm for Big Graph Data Clustering (Journal of Big Data 2016)**
  - Honglei Zhang, Jenni Raitoharju, Serkan Kiranyaz, and Moncef Gabbouj
  - [[Paper]](https://arxiv.org/abs/1606.06450)
  - [[C++ Reference]](https://github.com/harleyzhang/LRW)

Graph clustering is an important technique to understand the relationships between the vertices in a big graph. In this paper, we propose a novel random-walk-based graph clustering method. The proposed method restricts the reach of the walking agent using an inflation function and a normalization function. We analyze the behavior of the limited random walk procedure and propose a novel algorithm for both global and local graph clustering problems. Previous random-walk-based algorithms depend on the chosen fitness function to find the clusters around a seed vertex. The proposed algorithm tackles the problem in an entirely different manner. We use the limited random walk procedure to find attracting vertices in a graph and use them as features to cluster the vertices. According to the experimental results on the simulated graph data and the real-world big graph data, the proposed method is superior to the state-of-the-art methods in solving graph clustering problems. Since the proposed method uses the embarrassingly parallel paradigm, it can be efficiently implemented and embedded in any parallel computing environment such as a MapReduce framework. Given enough computing resources, we are capable of clustering graphs with millions of vertices and hundreds millions of edges in a reasonable time.

- **11.Community Detection Based on Structure and Content: A Content Propagation Perspective (ICDM 2015)**
  - Liyuan Liu, Linli Xu, Zhen Wang, and Enhong Chen 
  - [[Paper]](https://liyuanlucasliu.github.io/pdf/Liyuan-Liu-ICDM.pdf)
  - [[Matlab Reference]](https://github.com/LiyuanLucasLiu/Content-Propagation)
  
- **12.Modeling Community Detection Using Slow Mixing Random Walks (IEEE Big Data 2015)**
  - Ramezan Paravi, Torghabeh Narayana, and Prasad Santhanam
  - [[Paper]](https://ieeexplore.ieee.org/abstract/document/7364008)
  - [[Python Reference]](https://github.com/paravi/MarovCommunity)

The task of community detection in a graph formalizes the intuitive task of grouping together subsets of vertices such that vertices within clusters are connected tighter than those in disparate clusters. This paper approaches community detection in graphs by constructing Markov random walks on the graphs. The mixing properties of the random walk are then used to identify communities. We use coupling from the past as an algorithmic primitive to translate the mixing properties of the walk into revealing the community structure of the graph. We analyze the performance of our algorithms on specific graph structures, including the stochastic block models (SBM) and LFR random graphs.


- **13.GossipMap: A Distributed Community Detection Algorithm for Billion-Edge Directed Graphs (SC 2015)**
  - Seung-Hee Bae and Bill Howe
  - [[Paper]](https://dl.acm.org/citation.cfm?id=2807668)
  - [[C++ Reference]](https://github.com/uwescience/GossipMap)

In this paper, we describe a new distributed community detection algorithm for billion-edge directed graphs that, unlike modularity-based methods, achieves cluster quality on par with the best-known algorithms in the literature. We show that a simple approximation to the best-known serial algorithm dramatically reduces computation and enables distributed evaluation yet incurs only a very small impact on cluster quality.

We present three main results: First, we show that the clustering produced by our scalable approximate algorithm compares favorably with prior results on small synthetic benchmarks and small real-world datasets (70 million edges). Second, we evaluate our algorithm on billion-edge directed graphs (a 1.5B edge social network graph, and a 3.7B edge web crawl), and show that the results exhibit the structural properties predicted by analysis of much smaller graphs from similar sources. Third, we show that our algorithm exhibits over 90% parallel efficiency on massive graphs in weak scaling experiments.

- **14.Scalable Detection of Statistically Significant Communities and Hierarchies, Using Message Passing for Modularity (PNAS 2014)**
  - Pan Zhang and Cristopher Moore
  - [[Paper]](https://www.pnas.org/content/111/51/18144)
  - [[Python]](https://github.com/weberfm/belief_propagation_community_detection)

Modularity is a popular measure of community structure. However, maximizing the modularity can lead to many competing partitions, with almost the same modularity, that are poorly correlated with each other. It can also produce illusory ‘‘communities’’ in random graphs where none exist. We address this problem by using the modularity as a Hamiltonian at finite temperature and using an efficient belief propagation algorithm to obtain the consensus of many partitions with high modularity, rather than looking for a single partition that maximizes it. We show analytically and numerically that the proposed algorithm works all of the way down to the detectability transition in networks generated by the stochastic block model. It also performs well on real-world networks, revealing large communities in some networks where previous work has claimed no communities exist. Finally we show that by applying our algorithm recursively, subdividing communities until no statistically significant subcommunities can be found, we can detect hierarchical structure in real-world networks more efficiently than previous methods.

- **15.Efficient Monte Carlo and Greedy Heuristic for the Inference of Stochastic Block Models (Phys. Rev. E 2014)**
  - Tiago P. Peixoto
  - [[Paper]](https://arxiv.org/pdf/1310.4378)
  - [[Python Reference]](https://github.com/graphchallenge/GraphChallenge/tree/master/StochasticBlockPartition)

We present an efficient algorithm for the inference of stochastic block models in large networks.
The algorithm can be used as an optimized Markov chain Monte Carlo (MCMC) method, with
a fast mixing time and a much reduced susceptibility to getting trapped in metastable states, or
as a greedy agglomerative heuristic, with an almost linear O(N ln2 N) complexity, where N is the
number of nodes in the network, independent of the number of blocks being inferred. We show
that the heuristic is capable of delivering results which are indistinguishable from the more exact
and numerically expensive MCMC method in many artificial and empirical networks, despite being
much faster. The method is entirely unbiased towards any specific mixing pattern, and in particular
it does not favor assortative community structures.

- **16.Overlapping Community Detection Using Seed Set Expansion (CIKM 2013)**
  - Joyce Jiyoung Whang, David F. Gleich, and Inderjit S. Dhillon
  - [[Paper]](http://www.cs.utexas.edu/~inderjit/public_papers/overlapping_commumity_cikm13.pdf)
  - [[Python Reference]](https://github.com/pratham16/community-detection-by-seed-expansion)

Community detection is an important task in network analysis. A community (also referred to as a cluster) is a set
of cohesive vertices that have more connections inside the
set than outside. In many social and information networks,
these communities naturally overlap. For instance, in a social network, each vertex in a graph corresponds to an individual who usually participates in multiple communities.
One of the most successful techniques for finding overlapping
communities is based on local optimization and expansion
of a community metric around a seed set of vertices. In
this paper, we propose an efficient overlapping community
detection algorithm using a seed set expansion approach.
In particular, we develop new seeding strategies for a personalized PageRank scheme that optimizes the conductance
community score. The key idea of our algorithm is to find
good seeds, and then expand these seed sets using the personalized PageRank clustering procedure. Experimental results show that this seed set expansion approach outperforms other state-of-the-art overlapping community detection methods. We also show that our new seeding strategies
are better than previous strategies, and are thus effective in
finding good overlapping clusters in a graph.

- **17.Influence-Based Network-Oblivious Community Detection (ICDM 2013)**
  - Nicola Barbieri, Francesco Bonchi, and Giuseppe Manco 
  - [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6729581)
  - [[Java  Reference]](https://github.com/gmanco/cwn)
 
How can we detect communities when the social graphs is not available? We tackle this problem by modeling social contagion from a log of user activity, that is a dataset of tuples (u, i, t) recording the fact that user u "adopted" item i at time t. This is the only input to our problem. We propose a stochastic framework which assumes that item adoptions are governed by un underlying diffusion process over the unobserved social network, and that such diffusion model is based on community-level influence. By fitting the model parameters to the user activity log, we learn the community membership and the level of influence of each user in each community. This allows to identify for each community the "key" users, i.e., the leaders which are most likely to influence the rest of the community to adopt a certain item. The general framework can be instantiated with different diffusion models. In this paper we define two models: the extension to the community level of the classic (discrete time) Independent Cascade model, and a model that focuses on the time delay between adoptions. To the best of our knowledge, this is the first work studying community detection without the network.

- **18.SLPA: Uncovering Overlapping Communities in Social Networks via A Speaker-listener Interaction Dynamic Process (ICDMW 2011)**
  - Jierui Xie, Boleslaw K Szymanski, and Xiaoming Liu
  - [[Paper]](https://arxiv.org/pdf/1109.5720.pdf)
  - [[Java Reference]](https://github.com/sebastianliu/SLPA-community-detection)
  - [[Python Reference]](https://github.com/kbalasu/SLPA)
  - [[C++ Reference]](https://github.com/arminbalalaie/graphlab-slpa)

Overlap is one of the characteristics of social
networks, in which a person may belong to more than one social
group. For this reason, discovering overlapping structures
is necessary for realistic social analysis. In this paper, we
present a novel, general framework to detect and analyze
both individual overlapping nodes and entire communities. In
this framework, nodes exchange labels according to dynamic
interaction rules. A specific implementation called Speakerlistener Label Propagation Algorithm (SLPA1
) demonstrates
an excellent performance in identifying both overlapping nodes
and overlapping communities with different degrees of diversity.


- **19.On the Generation of Stable Communities of Users for Dynamic Mobile Ad Hoc Social Networks (IEEE ICOIN  2011)**
  - Guillaume-Jean Herbiet and Pascal Bouvry
  - [[Paper]](https://herbiet.gforge.uni.lu/research.html)
  - [[Java Reference]](https://github.com/gjherbiet/gs-sharc)
  
- **20.SHARC: Community-Based Partitioning for Mobile Ad Hoc Networks Using Neighborhood Similarity (IEEE WoWMoM 2010)**
  - Guillaume-Jean Herbiet and Pascal Bouvry
  - [[Paper]](https://herbiet.gforge.uni.lu/research.html)
  - [[Java Reference]](https://github.com/gjherbiet/gs-sharc)
  
- **21.Mapping Change in Large Networks (Plos One 2010)**
  - Rosvall M, Bergstrom 
  - [[Paper]](https://github.com/mapequation/significance-clustering)
  - [[C++ Reference]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0008694)
  
- **22.Graph Clustering Based on Structural/Attribute Similarities (WSDM 2009)**
  - Yang Zhou, Hong Cheng, Jeffrey Xu Yu
  - [[Paper]](http://www.vldb.org/pvldb/2/vldb09-175.pdf)
  - [[Python Reference]](https://github.com/zhanghuijun-hello/Graph-Clustering-Based-on-Structural-Attribute-Similarities-)

The goal of graph clustering is to partition vertices in a large graph
into different clusters based on various criteria such as vertex connectivity or neighborhood similarity. Graph clustering techniques
are very useful for detecting densely connected groups in a large
graph. Many existing graph clustering methods mainly focus on
the topological structure for clustering, but largely ignore the vertex properties which are often heterogenous. In this paper, we propose a novel graph clustering algorithm, SA-Cluster, based on both
structural and attribute similarities through a unified distance measure. Our method partitions a large graph associated with attributes
into k clusters so that each cluster contains a densely connected
subgraph with homogeneous attribute values. An effective method
is proposed to automatically learn the degree of contributions of
structural similarity and attribute similarity. Theoretical analysis is
provided to show that SA-Cluster is converging. Extensive experimental results demonstrate the effectiveness of SA-Cluster through
comparison with the state-of-the-art graph clustering and summarization methods.

- **23.Bridge Bounding: A Local Approach for Efficient Community Discovery in Complex Networks (ArXiv 2009)**
  - Symeon Papadopoulos, Andre Skusa, Athena Vakali, Yiannis Kompatsiaris, and Nadine Wagner
  - [[Paper]](https://arxiv.org/abs/0902.0871)
  - [[Java Reference]](https://github.com/kleinmind/bridge-bounding)
  
- **24.The Map Equation (The European Physical Journal Special Topics 2009)**
  - Martin Rossvall, Daniel Axelsson, and Carl T Bergstrom
  - [[Paper]](https://arxiv.org/abs/0906.1405)
  - [[R Reference]](igraph.org/r/doc/cluster_infomap.html)
  - [[C Reference]](http://igraph.org/c/)
  - [[Python Reference]](https://github.com/Tavpritesh/MapEquation)
  
- **25.Biclique Communities (Physical Review E  2008)**
  - Sune Lehmann, Martin Schwartz, and Lars Kai Hansen
  - [[Paper]](https://www.researchgate.net/publication/23230281_Biclique_communities)
  - [[R Reference]](https://github.com/hecking/bipartite_community_detection)
  
- **26.Near Linear Time Algorithm to Detect Community Structures in Large-Scale Networks (Physical Review E 2008)**
  - Usha Nandini Raghavan, Reka Albert, Soundar Kumara
  - [[Paper]](https://arxiv.org/abs/0709.2938)
  - [[Python Reference]](https://github.com/benedekrozemberczki/karateclub)
  - [[Python Reference]](https://github.com/benedekrozemberczki/LabelPropagation)
  - [[C++ Reference]](https://github.com/carlosmata/LabelPropagation)

- **27.Chinese Whispers: an Efficient Graph Clustering Algorithm and its Application to Natural Language Processing Problems (HLT NAACL 2006)**
  - Chris Biemann
  - [[Paper]](http://www.aclweb.org/anthology/W06-3812)
  - [[Python Reference]](https://github.com/sanmayaj/ChineseWhispers)
  - [[Python Alternative]](https://github.com/nlpub/chinese-whispers-python)
  
- **28.Uncovering the Overlapping Community Structure of Complex Networks in Nature and Society  (Nature 2005)**
  - Gergely Palla, Imre Derenyi, Illes Farkas, Tamas Vicsek
  - [[Paper]](https://www.researchgate.net/publication/7797121_Uncovering_the_overlapping_community_structure_of_complex_networks_in_nature_and_society)
  - [[Python Reference]](https://github.com/nhanwei/k_clique_percolation_spark)
  
- **29.An Efficient Algorithm for Large-scale Detection of Protein Families (Nucleic Acids Research 2002)**
  - Anton Enright, Stijn Van Dongen, and Christos Ouzounis
  - [[Paper]](https://academic.oup.com/nar/article/30/7/1575/2376029)
  - [[Python Reference]](https://github.com/HarshHarwani/markov-clustering-for-graphs)
  - [[Python Reference]](https://github.com/lucagiovagnoli/Markov_clustering-Graph_API)
