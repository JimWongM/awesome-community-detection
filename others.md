  
## Others

- **1.A Novel Density-based Clustering Algorithm Using Nearest Neighbor Graph (Pattern Recognition 2020)**
  - Hao Li, Xiaojie Liu, Tao Li, and Rundong Gan
  - [[Paper]](https://www.sciencedirect.com/science/article/abs/pii/S0031320320300121?via%3Dihub)
  - [[Python Reference]](https://github.com/tommylee3003/SDBSCAN)

Density-based clustering has several desirable properties, such as the abilities to handle and identify noise samples, discover clusters of arbitrary shapes, and automatically discover of the number of clusters. Identifying the core samples within the dense regions of a dataset is a significant step of the density-based clustering algorithm. Unlike many other algorithms that estimate the density of each samples using different kinds of density estimators and then choose core samples based on a threshold, in this paper, we present a novel approach for identifying local high-density samples utilizing the inherent properties of the nearest neighbor graph (NNG). After using the density estimator to filter noise samples, the proposed algorithm ADBSCAN in which “A” stands for “Adaptive” performs a DBSCAN-like clustering process. The experimental results on artificial and real-world datasets have demonstrated the significant performance improvement over existing density-based clustering algorithms.

- **2.Ensemble Clustering for Graphs: Comparisons and Applications (Applied Network Science 2019)**
  - Valérie Poulin and François Théberge
  - [[Paper]](https://arxiv.org/abs/1809.05578)
  - [[Python Reference]](https://github.com/ftheberge/Ensemble-Clustering-for-Graphs)

We propose an ensemble clustering algorithm for graphs (ECG), which is based on the Louvain algorithm and the concept of consensus clustering. We validate our approach by replicating a recently published study comparing graph clustering algorithms over artificial networks, showing that ECG outperforms the leading algorithms from that study. We also illustrate how the ensemble obtained with ECG can be used to quantify the presence of community structure in the graph.

- **3.CutESC: Cutting Edge Spatial Clustering Technique based on Proximity Graphs (Knowledge-Based Systems 2019)**
  - Alper Aksac, Tansel Özyer, Reda Alhajja
  - [[Paper]](https://www.sciencedirect.com/science/article/pii/S0031320319302468)
  - [[Python Reference]](https://github.com/alperaksac/cutESC)

In this paper, we propose a cut-edge algorithm for spatial clustering (CutESC) based on proximity graphs. The CutESC algorithm removes edges when a cut-edge value for the edge’s endpoints is below a threshold. The cut-edge value is calculated by using statistical features and spatial distribution of data based on its neighborhood. Also, the algorithm works without any prior information and preliminary parameter settings while automatically discovering clusters with non-uniform densities, arbitrary shapes, and outliers. However, there is an option which allows users to set two parameters to better adapt clustering solutions for particular problems. To assess advantages of CutESC algorithm, experiments have been conducted using various two-dimensional synthetic, high-dimensional real-world, and image segmentation datasets.

- **4.A Study of Graph-based System for Multi-view Clustering (Knowledge-Based Systems 2019)**
  - Hao Wang, Yan Yang, BingLiu, Hamido Fujita
  - [[Paper]](https://www.sciencedirect.com/science/article/pii/S0950705118305082)
  - [[Python Reference]](https://github.com/cswanghao/gbs)

This paper studies clustering of multi-view data, known as multi-view clustering. Among existing multi-view clustering methods, one representative category of methods is the graph-based approach. Despite its elegant and simple formulation, the graph-based approach has not been studied in terms of (a) the generalization of the approach or (b) the impact of different graph metrics on the clustering results. This paper extends this important approach by first proposing a general Graph-Based System (GBS) for multi-view clustering, and then discussing and evaluating the impact of different graph metrics on the multi-view clustering performance within the proposed framework. GBS works by extracting data feature matrix of each view, constructing graph matrices of all views, and fusing the constructed graph matrices to generate a unified graph matrix, which gives the final clusters. A novel multi-view clustering method that works in the GBS framework is also proposed, which can (1) construct data graph matrices effectively, (2) weight each graph matrix automatically, and (3) produce clustering results directly. Experimental results on benchmark datasets show that the proposed method outperforms state-of-the-art baselines significantly.

- **5.Learning Resolution Parameters for Graph Clustering (WWW 2019)**
  - Nate Veldt, David Gleich, Anthony Wirth
  - [[Paper]](https://arxiv.org/abs/1903.05246)
  - [[Julia Reference]](https://github.com/nveldt/LearnResParams)

Finding clusters of well-connected nodes in a graph is an extensively studied problem in graph-based data analysis. Because of its many applications, a large number of distinct graph clustering objective functions and algorithms have already been proposed and analyzed. To aid practitioners in determining the best clustering approach to use in different applications, we present new techniques for automatically learning how to set clustering resolution parameters. These parameters control the size and structure of communities that are formed by optimizing a generalized objective function. We begin by formalizing the notion of a parameter fitness function, which measures how well a fixed input clustering approximately solves a generalized clustering objective for a specific resolution parameter value. Under reasonable assumptions, which suit two key graph clustering applications, such a parameter fitness function can be efficiently minimized using a bisection-like method, yielding a resolution parameter that fits well with the example clustering. We view our framework as a type of single-shot hyperparameter tuning, as we are able to learn a good resolution parameter with just a single example. Our general approach can be applied to learn resolution parameters for both local and global graph clustering objectives. We demonstrate its utility in several experiments on real-world data where it is helpful to learn resolution parameters from a given example clustering.

- **6.Multiview Consensus Graph Clustering (IEEE TIP 2019)**
  - Kun Zhan and Feiping Nie and Jing Wang and Yi Yang
  - [[Paper]](https://www.ncbi.nlm.nih.gov/pubmed/30346283)
  - [[Matlab Reference]](https://github.com/kunzhan/MCGC)

A graph is usually formed to reveal the relationship between data points and graph structure is encoded by the affinity matrix. Most graph-based multiview clustering methods use predefined affinity matrices and the clustering performance highly depends on the quality of graph. We learn a consensus graph with minimizing disagreement between different views and constraining the rank of the Laplacian matrix. Since diverse views admit the same underlying cluster structure across multiple views, we use a new disagreement cost function for regularizing graphs from different views toward a common consensus. Simultaneously, we impose a rank constraint on the Laplacian matrix to learn the consensus graph with exactly connected components where is the number of clusters, which is different from using fixed affinity matrices in most existing graph-based methods. With the learned consensus graph, we can directly obtain the cluster labels without performing any post-processing, such as -means clustering algorithm in spectral clustering-based methods. A multiview consensus clustering method is proposed to learn such a graph. An efficient iterative updating algorithm is derived to optimize the proposed challenging optimization problem. Experiments on several benchmark datasets have demonstrated the effectiveness of the proposed method in terms of seven metrics.

- **7.CutESC: Cutting Edge Spatial Clustering Technique based on Proximity Graphs (Knowledge-Based Systems 2019)**
  - Alper Aksac, Tansel Özyer, Reda Alhajja
  - [[Paper]](https://www.sciencedirect.com/science/article/pii/S0031320319302468)
  - [[Python Reference]](https://github.com/alperaksac/cutESC)
  
- **8.Clubmark - Bench bencmarking Framework for the Clustering Algorithms Evaluation (ICDM 2018)**
  - Artem Lutov, Mourad Khayati, Philippe Cudre-Mauroux
  - [[Paper]](https://github.com/eXascaleInfolab/clubmark/blob/master/docs/clubmark.pdf)
  - [[Python Reference]](https://github.com/eXascaleInfolab/clubmark/tree/master/algorithms)


![image](https://raw.githubusercontent.com/JimWongM/ImageHost/master/img/20200408140529.png)

- **9.The Difference Between Optimal and Germane Communities (Social Network Analysis and Mining 2018)**
  - Jerry Scripps, Christian TrefftzZachary Kurmas
  - [[Paper]](https://link.springer.com/article/10.1007/s13278-018-0522-1)
  - [[Java Reference]](https://cis.gvsu.edu/~scrippsj/pubs/software.htm)

Networks often exhibit community structure and there are many algorithms that have been proposed to detect the communities. Different sets of communities have different characteristics. Community finding algorithms that are designed to optimize a single statistic tend to detect communities with a narrow set of characteristics. In this paper, we present evidence for the differences in community characteristics. In addition, we present two new community finding algorithms that allow analysts to find community sets that are not only high quality but also germane to the characteristics that are desired.

- **10.Discovering Fuzzy Structural Patterns for Graph Analytics (IEEE TFS 2018)**
  - Tiantian He  and Keith C. C. Chan 
  - [[Paper]](https://ieeexplore.ieee.org/document/8253904)
  - [[Executable Reference]](https://github.com/he-tiantian/FSPGA)

Many real-world datasets can be represented as attributed graphs that contain vertices, each of which is associated with a set of attribute values. Discovering clusters, or communities, which are structural patterns in these graphs, are one of the most important tasks in graph analysis. To perform the task, a number of algorithms have been proposed. Some of them detect clusters of particular topological properties, whereas some others discover them mainly based on attribute information. Also, most of the algorithms discover disjoint clusters only. As a result, they may not be able to detect more meaningful clusters hidden in the attributed graph. To do so more effectively, we propose an algorithm, called FSPGA, to discover fuzzy structural patterns for graph analytics. FSPGA performs the task of cluster discovery as a fuzzy-constrained optimization problem, which takes into consideration both the graph topology and attribute values. FSPGA has been tested with both synthetic and real-world graph datasets and is found to be efficient and effective at detecting clusters in attributed graphs. FSPGA is a promising fuzzy algorithm for structural pattern detection in attributed graphs.

- **11.Wiring Together Large Single-Cell RNA-Seq Sample Collections (biorxiv 2018)**
  - Nikolas Barkas, Viktor Petukhov, Daria Nikolaeva, Yaroslav Lozinsky, Samuel Demharter, Konstantin Khodosevich, Peter V. Kharchenko
  - [[Paper]](https://www.biorxiv.org/content/10.1101/460246v1)
  - [[C++]](https://github.com/hms-dbmi/conos)

Single-cell RNA-seq methods are being increasingly applied in complex study designs, which involve measurements of many samples, commonly spanning multiple individuals, conditions, or tissue compartments. Joint analysis of such extensive, and often heterogeneous, sample collections requires a way of identifying and tracking recurrent cell subpopulations across the entire collection. Here we describe a flexible approach, called Conos (Clustering On Network Of Samples), that relies on multiple plausible inter-sample mappings to construct a global graph connecting all measured cells. The graph can then be used to propagate information between samples and to identify cell communities that show consistent grouping across broad subsets of the collected samples. Conos results enable investigators to balance between resolution and breadth of the detected subpopulations. In this way, it is possible to focus on the fine-grained clusters appearing within more similar subsets of samples, or analyze coarser clusters spanning broader sets of samples in the collection. Such multi-resolution joint clustering provides an important basis for downstream analysis and interpretation of sizeable multi-sample single-cell studies and atlas-scale collections.

- **12.Community Detection and Stochastic Block Models: Recent Developments (JMLR 2017)**
  - Emmanuel Abbe
  - [[Paper]](https://arxiv.org/pdf/1703.10146v1.pdf)
  
- **13.Watset: Automatic Induction of Synsets for a Graph of Synonyms (ACL 2017)**
  - Dmitry Ustalov, Alexander Panchenko, and Chris Biemann
  - [[Paper]](https://doi.org/10.18653/v1/P17-1145)
  - [[Python Reference]](https://github.com/dustalov/watset)
  - [[Java Reference]](https://github.com/nlpub/watset-java)

This paper presents a new graph-based approach that induces synsets using synonymy dictionaries and word embeddings. First, we build a weighted graph of synonyms extracted from commonly available resources, such as Wiktionary. Second, we apply word sense induction to deal with ambiguous words. Finally, we cluster the disambiguated version of the ambiguous input graph into synsets. Our meta-clustering approach lets us use an efficient hard clustering algorithm to perform a fuzzy clustering of the graph. Despite its simplicity, our approach shows excellent results, outperforming five competitive state-of-the-art methods in terms of F-score on three gold standard datasets for English and Russian derived from large-scale manually constructed lexical resources.

- **14.An Overlapping Community Detection Algorithm Based on Density Peaks (NeuroComputing 2017)**
  - Xueying Bai, Peilin Yang, and Xiaohu Shi
  - [[Paper]](https://www.sciencedirect.com/science/article/pii/S092523121631400X)
  - [[Matlab Reference]](https://github.com/XueyingBai/An-overlapping-community-detection-algorithm-based-on-density-peaks)

Many real-world networks contain overlapping communities like protein-protein networks and social networks. Overlapping community detection plays an important role in studying hidden structure of those networks. In this paper, we propose a novel overlapping community detection algorithm based on density peaks (OCDDP). OCDDP utilizes a similarity based method to set distances among nodes, a three-step process to select cores of communities and membership vectors to represent belongings of nodes. Experiments on synthetic networks and social networks prove that OCDDP is an effective and stable overlapping community detection algorithm. Compared with the top existing methods, it tends to perform better on those “simple” structure networks rather than those infrequently “complicated” ones.

- **15.Fast Heuristic Algorithm for Multi-scale Hierarchical Community Detection (ASONAM 2017)**
  - Eduar Castrillo, Elizabeth León, and Jonatan Gómez
  - [[Paper]](https://arxiv.org/abs/1707.02362)
  - [[C++ Reference]](https://github.com/eduarc/HAMUHI)

Complex networks constitute the backbones of many complex systems such as social networks. Detecting the community structure in a complex network is both a challenging and a computationally expensive task. In this paper, we present the HAMUHI-CODE, a novel fast heuristic algorithm for multi-scale hierarchical community detection inspired on an agglomerative hierarchical clustering technique. We define a new structural similarity of vertices based on the classical cosine similarity by removing some vertices in order to increase the probability of identifying inter-cluster edges. Then we use the proposed structural similarity in a new agglomerative hierarchical algorithm that does not merge only clusters with maximal similarity as in the classical approach, but merges any cluster that does not meet a parameterized community definition with its most similar adjacent cluster. The algorithm computes all the similar clusters at the same time is checking if each cluster meets the parameterized community definition. It is done in linear time complexity in terms of the number of cluster in the iteration. Since a complex network is a sparse graph, our approach HAMUHI-CODE has a super-linear time complexity with respect to the size of the input in the worst-case scenario (if the clusters merge in pairs), making it suitable to be applied on large-scale complex networks. To test the properties and the efficiency of our algorithm we have conducted extensive experiments on real world and synthetic benchmark networks by comparing it to several baseline state-of-the-art algorithms.

- **16.Time Series Clustering via Community Detection in Networks (Information Sciences 2016)**
  - Leonardo N. Ferreira and Liang Zhao
  - [[Paper]](https://www.sciencedirect.com/science/article/pii/S002002551500554X)
  - [[R Reference]](https://github.com/lnferreira/time_series_clustering_via_community_detection)

In this paper, we propose a technique for time series clustering using community detection in complex networks. Firstly, we present a method to transform a set of time series into a network using different distance functions, where each time series is represented by a vertex and the most similar ones are connected. Then, we apply community detection algorithms to identify groups of strongly connected vertices (called a community) and, consequently, identify time series clusters. Still in this paper, we make a comprehensive analysis on the influence of various combinations of time series distance functions, network generation methods and community detection techniques on clustering results. Experimental study shows that the proposed network-based approach achieves better results than various classic or up-to-date clustering techniques under consideration. Statistical tests confirm that the proposed method outperforms some classic clustering algorithms, such as k-medoids, diana, median-linkage and centroid-linkage in various data sets. Interestingly, the proposed method can effectively detect shape patterns presented in time series due to the topological structure of the underlying network constructed in the clustering process. At the same time, other techniques fail to identify such patterns. Moreover, the proposed method is robust enough to group time series presenting similar pattern but with time shifts and/or amplitude variations. In summary, the main point of the proposed method is the transformation of time series from time-space domain to topological domain. Therefore, we hope that our approach contributes not only for time series clustering, but also for general time series analysis tasks.


- **17.Community Detection in Multi-Partite Multi-Relational Networks Based on Information Compression (New Generation Computing 2016)**
  - Xin Liu, Weichu Liu, Tsuyoshi Murata, and Ken Wakita
  - [[Paper]](https://link.springer.com/article/10.1007/s00354-016-0206-1)
  - [[Scala Reference]](https://github.com/weichuliu/hetero_scala)

Community detection in uni-partite single-relational networks which contain only one type of nodes and edges has been extensively studied in the past decade. However, many real-world systems are naturally described as multi-partite multi-relational networks which contain multiple types of nodes and edges. In this paper, we propose an information compression based method for detecting communities in such networks. Specifically, based on the minimum description length (MDL) principle, we propose a quality function for evaluating partitions of a multi-partite multi-relational network into communities, and develop a heuristic algorithm for optimizing the quality function. We demonstrate that our method outperforms the state-of-the-art techniques in both synthetic and real-world networks.


- **18.Integration of Graph Clustering with Ant Colony Optimization for Feature Selection (Knowledge-Based Systems 2015)**
  - Parham Moradi, Mehrdad Rostami
  - [[Paper]](http://www.sciencedirect.com/science/article/pii/S0950705115001458)
  - [[Matlab Reference]](https://github.com/XuesenYang/Graph-clustering-with-ant-colony-optimization-for-feature-selection)  

Feature selection is an important preprocessing step in machine learning and pattern recognition. The ultimate goal of feature selection is to select a feature subset from the original feature set to increase the performance of learning algorithms. In this paper a novel feature selection method based on the graph clustering approach and ant colony optimization is proposed for classification problems. The proposed method’s algorithm works in three steps. In the first step, the entire feature set is represented as a graph. In the second step, the features are divided into several clusters using a community detection algorithm and finally in the third step, a novel search strategy based on the ant colony optimization is developed to select the final subset of features. Moreover the selected subset of each ant is evaluated using a supervised filter based method called novel separability index. Thus the proposed method does not need any learning model and can be classified as a filter based feature selection method. The proposed method integrates the community detection algorithm with a modified ant colony based search process for the feature selection problem. Furthermore, the sizes of the constructed subsets of each ant and also size of the final feature subset are determined automatically. The performance of the proposed method has been compared to those of the state-of-the-art filter and wrapper based feature selection methods on ten benchmark classification problems. The results show that our method has produced consistently better classification accuracies.

- **19.Greedy Discrete Particle Swarm Optimization for Large-Scale Social Network Clustering (Information Sciences 2015)**
  - Qing Cai, Maoguo Gong, Lijia Ma, Shasha Ruan, Fuyan Yuan, Licheng Jiao
  - [[Paper]](https://www.sciencedirect.com/science/article/pii/S0020025514009530)
  - [[C++ Reference]](https://github.com/doctor-cai/GDPSO)

Social computing is a new paradigm for information and communication technology. Social network analysis is one of the theoretical underpinnings of social computing. Community structure detection is believed to be an effective tool for social network analysis. Uncovering community structures in social networks can be regarded as clustering optimization problems. Because social networks are characterized by dynamics and huge volumes of data, conventional nature-inspired optimization algorithms encounter serious challenges when applied to solving large-scale social network clustering optimization problems. In this study, we put forward a novel particle swarm optimization algorithm to reveal community structures in social networks. The particle statuses are redefined under a discrete scenario. The status updating rules are reconsidered based on the network topology. A greedy strategy is designed to drive particles to a promising region. To this end, a greedy discrete particle swarm optimization framework for large-scale social network clustering is suggested. To determine the performance of the algorithm, extensive experiments on both synthetic and real-world social networks are carried out. We also compare the proposed algorithm with several state-of-the-art network community clustering methods. The experimental results demonstrate that the proposed method is effective and promising for social network clustering.

- **20.Community Detection via Maximization of Modularity and Its Variants (IEEE TCSS 2014)**
  - Mingming Chen, Konstantin Kuzmin, and Boleslaw K. Szymanski 
  - [[Paper]](https://www.cs.rpi.edu/~szymansk/papers/TCSS-14.pdf)
  - [[Python Reference]](https://github.com/itaneja2/community-detection)


![image](https://raw.githubusercontent.com/JimWongM/ImageHost/master/img/20200408142157.png)

- **21.A Smart Local Moving Algorithm for Large-Scale Modularity-Based Community Detection (The European Physical Journal B 2013)**
  - Ludo Waltman and Nees Jan Van Eck
  - [[Paper]](https://link.springer.com/content/pdf/10.1140/epjb/e2013-40829-0.pdf)
  - [[R Reference]](https://github.com/chen198328/slm)
  - [[Python Reference]](https://github.com/iosonofabio/slmpy)

We introduce a new algorithm for modularity-based community detection in large networks. The algorithm, which we refer to as a smart local moving algorithm, takes advantage of a well-known local moving heuristic that is also used by other algorithms. Compared with these other algorithms, our proposed algorithm uses the local moving heuristic in a more sophisticated way. Based on an analysis of a diverse set of networks, we show that our smart local moving algorithm identifies community structures with higher modularity values than other algorithms for large-scale modularity optimization, among which the popular “Louvain algorithm”. The computational efficiency of our algorithm makes it possible to perform community detection in networks with tens of millions of nodes and hundreds of millions of edges. Our smart local moving algorithm also performs well in small and medium-sized networks. In short computing times, it identifies community structures with modularity values equally high as, or almost as high as, the highest values reported in the literature, and sometimes even higher than the highest values found in the literature.

- **22.Bayesian Hierarchical Community Discovery (NIPS 2013)**
  - Charles Blundell and Yee Whye Teh
  - [[Paper]](http://papers.nips.cc/paper/5048-bayesian-hierarchical-community-discovery.pdf)
  - [[Python Reference]](https://github.com/krzychu/bhcd/)
  - [[C++ Reference]](https://github.com/blundellc/bhcd/)

We propose an efficient Bayesian nonparametric model for discovering hierarchical community structure in social networks. Our model is a tree-structured
mixture of potentially exponentially many stochastic blockmodels. We describe a
family of greedy agglomerative model selection algorithms that take just one pass
through the data to learn a fully probabilistic, hierarchical community model. In
the worst case, Our algorithms scale quadratically in the number of vertices of
the network, but independent of the number of nested communities. In practice,
the run time of our algorithms are two orders of magnitude faster than the Infinite
Relational Model, achieving comparable or better accuracy.

- **23.Efficient Discovery of Overlapping Communities in Massive Networks (PNAS 2013)**
  - Prem K. Gopalan and David M. Blei
  - [[Paper]](https://www.pnas.org/content/110/36/14534)
  - [[C++ Reference]](https://github.com/premgopalan/svinet)
  
- **24.Complex Network Clustering by Multiobjective Discrete Particle Swarm Optimization Based on Decomposition (IEEE Trans. Evolutionary Computation 2013)**
  - Maoguo Gong, Qing Cai, Xiaowei Chen, and Lijia Ma
  - [[Paper]](https://ieeexplore.ieee.org/document/6510542?reason=concurrency)
  - [[C++ Reference]](https://github.com/doctor-cai/MODPSO)
  
- **25.An Efficient and Principled Method for Detecting Communities in Networks (Physical Review E 2011)**
  - Brian Ball, Brian Karrer, M. E. J. Newman
  - [[Paper]](https://arxiv.org/pdf/1104.3590.pdf)
  - [[C++ Reference]](http://www.personal.umich.edu/~mejn/OverlappingLinkCommunities.zip)
  - [[Python Reference]](https://github.com/Zabot/principled_clustering)
  
- **26.A Game-Theoretic Approach to Hypergraph Clustering (NIPS 2009)**
  - Samuel R. Bulò and Marcello Pelillo
  - [[Paper]](https://papers.nips.cc/paper/3714-a-game-theoretic-approach-to-hypergraph-clustering)
  - [[Matlab Reference]](https://github.com/Schofield-Mao/Gametheory-Clustring)
