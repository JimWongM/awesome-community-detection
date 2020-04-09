## Cyclic Patterns

- **1.DAOC: Stable Clustering of Large Networks (Phys. Rev E 2019)**
  - Artem Lutov, Mourad Khayati, Philippe Cudré-Mauroux
  - [[Paper]](https://arxiv.org/abs/1909.08786)
  - [[C++ Reference]](https://github.com/eXascaleInfolab/daoc)

Clustering is a crucial component of many data mining systems involving the analysis and exploration of various data. Data diversity calls for clustering algorithms to be accurate while providing stable (i.e., deterministic and robust) results on arbitrary input networks. Moreover, modern systems often operate with large datasets, which implicitly constrains the complexity of the clustering algorithm. Existing clustering techniques are only partially stable, however, as they guarantee either determinism or robustness. To address this issue, we introduce DAOC, a Deterministic and Agglomerative Overlapping Clustering algorithm. DAOC leverages a new technique called Overlap Decomposition to identify fine-grained clusters in a deterministic way capturing multiple optima. In addition, it leverages a novel consensus approach, Mutual Maximal Gain, to ensure robustness and further improve the stability of the results while still being capable of identifying micro-scale clusters. Our empirical results on both synthetic and real-world networks show that DAOC yields stable clusters while being on average 25% more accurate than state-of-the-art deterministic algorithms without requiring any tuning. Our approach has the ambition to greatly simplify and speed up data analysis tasks involving iterative processing (need for determinism) as well as data fluctuations (need for robustness) and to provide accurate and reproducible results.

- **2.Fast Consensus Clustering in Complex Networks (Phys. Rev E 2019)**
  - Aditya Tandon, Aiiad Albeshri, Vijey Thayananthan, Wadee Alhalabi, Santo Fortunato
  - [[Paper]](https://arxiv.org/abs/1902.04014)
  - [[Python Reference]](https://github.com/adityat/fastconsensus)

Algorithms for community detection are usually stochastic, leading to different partitions for different choices of random seeds. Consensus clustering has proven to be an effective technique to derive more stable and accurate partitions than the ones obtained by the direct application of the algorithm. However, the procedure requires the calculation of the consensus matrix, which can be quite dense if (some of) the clusters of the input partitions are large. Consequently, the complexity can get dangerously close to quadratic, which makes the technique inapplicable on large graphs. Here we present a fast variant of consensus clustering, which calculates the consensus matrix only on the links of the original graph and on a comparable number of additional node pairs, suitably chosen. This brings the complexity down to linear, while the performance remains comparable as the full technique. Therefore, our fast consensus clustering procedure can be applied on networks with millions of nodes and links.

- **3.EdMot: An Edge Enhancement Approach for Motif-aware Community Detection (KDD 2019)**
  - Pei-Zhen Li, Ling Huang, Chang-Dong Wang, and Jian-Huang Lai 
  - [[Paper]](https://arxiv.org/abs/1906.04560)
  - [[Python Reference]](https://github.com/benedekrozemberczki/EdMot)
  - [[Matlab Reference]](https://github.com/lipzh5/EdMot_pro)

Network community detection is a hot research topic in network analysis. Although many methods have been proposed for community detection, most of them only take into consideration the lower-order structure of the network at the level of individual nodes and edges. Thus, they fail to capture the higher-order characteristics at the level of small dense subgraph patterns, e.g., motifs. Recently, some higher-order methods have been developed but they typically focus on the motif-based hypergraph which is assumed to be a connected graph. However, such assumption cannot be ensured in some real-world networks. In particular, the hypergraph may become fragmented. That is, it may consist of a large number of connected components and isolated nodes, despite the fact that the original network is a connected graph. Therefore, the existing higher-order methods would suffer seriously from the above fragmentation issue, since in these approaches, nodes without connection in hypergraph can't be grouped together even if they belong to the same community. To address the above fragmentation issue, we propose an Edge enhancement approach for Motif-aware community detection (EdMot). The main idea is as follows. Firstly, a motif-based hypergraph is constructed and the top K largest connected components in the hypergraph are partitioned into modules. Afterwards, the connectivity structure within each module is strengthened by constructing an edge set to derive a clique from each module. Based on the new edge set, the original connectivity structure of the input network is enhanced to generate a rewired network, whereby the motif-based higher-order structure is leveraged and the hypergraph fragmentation issue is well addressed. Finally, the rewired network is partitioned to obtain the higher-order community structure.

- **4.From Louvain to Leiden: Guaranteeing Well-connected Communities (Scientific Reports 2019)**
  - Vincent Traag, Ludo Waltman, Nees Jan van Eck
  - [[Paper]](https://arxiv.org/abs/1810.08473)
  - [[C++ Reference]](https://github.com/vtraag/leidenalg)
  - [[Julia Reference]](https://github.com/bicycle1885/Leiden.jl)

Community detection is often used to understand the structure of large and complex networks. One of the most popular algorithms for uncovering community structure is the so-called Louvain algorithm. We show that this algorithm has a major defect that largely went unnoticed until now: the Louvain algorithm may yield arbitrarily badly connected communities. In the worst case, communities may even be disconnected, especially when running the algorithm iteratively. In our experimental analysis, we observe that up to 25% of the communities are badly connected and up to 16% are disconnected. To address this problem, we introduce the Leiden algorithm. We prove that the Leiden algorithm yields communities that are guaranteed to be connected. In addition, we prove that, when the Leiden algorithm is applied iteratively, it converges to a partition in which all subsets of all communities are locally optimally assigned. Furthermore, by relying on a fast local move approach, the Leiden algorithm runs faster than the Louvain algorithm. We demonstrate the performance of the Leiden algorithm for several benchmark and real-world networks. We find that the Leiden algorithm is faster than the Louvain algorithm and uncovers better partitions, in addition to providing explicit guarantees.

- **5.Anti-community Detection in Complex Networks (SSDBM 2018)**
  - Sebastian Lackner, Andreas Spitz, Matthias Weidemüller and Michael Gertz
  - [[Paper]](https://dbs.ifi.uni-heidelberg.de/files/Team/slackner/publications/Lackner_et_al_2018_Anti_Communities.pdf)
  - [[C Reference]](https://github.com/slackner/anti-community/)


Modeling the relations between the components of complex systems as networks of vertices and edges is a commonly used method in many scientific disciplines that serves to obtain a deeper understanding of the systems themselves. In particular, the detection of densely connected communities in these networks is frequently used to identify functionally related components, such as social circles in networks of personal relations or interactions between agents in biological networks. Traditionally, communities are considered to have a high density of internal connections, combined with a low density of external edges between different communities. However, not all naturally occurring communities in complex networks are characterized by this notion of structural equivalence, such as groups of energy states with shared quantum numbers in networks of spectral line transitions. In this paper, we focus on this inverse task of detecting anti-communities that are characterized by an exceptionally low density of internal connections and a high density of external connections. While anti-communities have been discussed in the literature for anecdotal applications or as a modification of traditional community detection, no rigorous investigation of algorithms for the problem has been presented. To this end, we introduce and discuss a broad range of possible approaches and evaluate them with regard to efficiency and effectiveness on a range of real-world and synthetic networks. Furthermore, we show that the presence of a community and anti-community structure are not mutually exclusive, and that even networks with a strong traditional community structure may also contain anti-communities.

- **6.Adaptive Modularity Maximization via Edge Weighting Scheme (Information Sciences 2018)**
  - Xiaoyan Lu, Konstantin Kuzmin, Mingming Chen, and Boleslaw K Szymanski
  - [[Paper]](https://link.springer.com/chapter/10.1007/978-3-319-72150-7_23)
  - [[Python Reference]](https://github.com/xil12008/adaptive_modularity)
  
Modularity maximization is one of the state-of-the-art methods for community detection that has gained popularity in the last decade. Yet it suffers from the resolution limit problem by preferring under certain conditions large communities over small ones. To solve this problem, we propose to expand the meaning of the edges that are currently used to indicate propensity of nodes for sharing the same community. In our approach this is the role of edges with positive weights while edges with negative weights indicate aversion for putting
their end-nodes into one community. We also present a novel regression model which assigns weights to the edges of a graph according to their local topological features to enhance the accuracy of modularity maximization algorithms. We construct artificial graphs based on the parameters sampled from a given unweighted network and train the regression model on ground truth communities of these artificial graphs in a supervised fashion. The extraction of local topological edge features can be done in linear time, making this process efficient.Experimental results on real and synthetic networks show that the state-of-theart community detection algorithms improve their performance significantly by finding communities in the weighted graphs produced by our model

- **7.Semi-Supervised Community Detection Using Structure and Size (ICDM 2018)**
  - Arjun Bakshi, Srinivasan Parthasarathy, and Kannan Srinivasan
  - [[Paper]](!!Missing!!)
  - [[Python Reference]](https://github.com/abaxi/bespoke-icdm18)
  - [[Python]](https://github.com/yzhang1918/bespoke-sscd)
  
- **8.Graph Sketching-based Space-efficient Data Clustering (SDM 2018)**
  - Anne Morvan, Krzysztof Choromanski, Cédric Gouy-Pailler, Jamal Atif
  - [[Paper]](https://epubs.siam.org/doi/pdf/10.1137/1.9781611975321.2)
  - [[Python Reference]](https://github.com/annemorvan/DBMSTClu)
  
  ![image](https://raw.githubusercontent.com/JimWongM/ImageHost/master/img/20200406150722.png)

- **9.Hierarchical Graph Clustering using Node Pair Sampling (Arxiv 2018)**
  - Thomas Bonald, Bertrand Charpentier, Alexis Galland and Alexandre Hollocou
  - [[Paper]](https://arxiv.org/pdf/1806.01664v2.pdf)
  - [[Python Reference]](https://github.com/tbonald/paris)
  
We present a novel hierarchical graph clustering algorithm inspired by modularity-based clustering
techniques. The algorithm is agglomerative and based on a simple distance between clusters induced by
the probability of sampling node pairs. We prove that this distance is reducible, which enables the use
of the nearest-neighbor chain to speed up the agglomeration. The output of the algorithm is a regular
dendrogram, which reveals the multi-scale structure of the graph. The results are illustrated on both
synthetic and real datasets.

- **10.Expander Decomposition and Pruning: Faster, Stronger, and Simpler (Arxiv 2018)**
  - Thatchaphol Saranurak, Di Wang
  - [[Paper]](https://arxiv.org/abs/1812.08958)
  - [[C++ Reference]](https://github.com/Skantz/expander-decomposition)
  
- **11.Priority Based Clustering in Weighted Graph Streams (JISE 2018)**
  - Mohsen Saadatpour, Sayyed Kamyar Izadi, Mohammad Nasirifar, and Hamed Kavoosi
  - [[Paper]](https://www.researchgate.net/publication/326622737_Priority-based_clustering_in_weighted_graph_streams)
  - [[Java Reference]](https://github.com/farnasirim/pri-bas-clu)
  
Nowadays, analyzing social networks is one of interesting research issues. Each network could be modeled by a graph structure. Clustering the vertices of this graph is a proper method to analyze the network. However, huge amount of changes in the graph structure as a result of social network interactions implies the need of an efficient clustering algorithm to process the stream of updates in a real-time manner. In this paper, we propose a novel algorithm for dynamic networks clustering based on the stream model. In our proposed algorithm, called Priority-based Clustering of Weighted Graph Streams (PCWGS), we provide a measure based on the importance of the frequency of recent interactions in the network to have more acceptable clusters. In PCWGS algorithm, a timestamp coupled with the weighted mean of the number of interactions of the network vertices are used to account edge weights. It is worth noting that, we present a data structure, which can keep useful information about the current state of the edges in the network based on update times and their weights while minimizing the required memory space in our proposed algorithm. Our simulations on real data sets reveal that PCWGS algorithm yields clustering with high quality and performance compared to previous state-of-the-art evolution-aware clustering algorithms.

- **12.Graph Learning for Multiview Clustering (IEEE Transactions on Cybernetics 2017)**
  - Anne Morvan, Krzysztof Choromanski, Cédric Gouy-Pailler, and Jamal Atif
  - [[Paper]](https://link.springer.com/chapter/10.1007/978-3-319-72150-7_23)
  - [[Matlab Reference]](https://github.com/kunzhan/MVGL)

 This study proposes ComSim, a new algorithm to detect communities in bipartite networks. This approach generates a partition of   ⊤  nodes by relying on similarity between the nodes in terms of links towards   ⊥  nodes. In order to show the relevance of this approach, we implemented and tested the algorithm on 2 small datasets equipped with a ground-truth partition of the nodes. It turns out that, compared to 3 baseline algorithms used in the context of bipartite graph, ComSim proposes the best communities. In addition, we tested the algorithm on a large scale network. Results show that ComSim has good performances, close in time to Louvain. Besides, a qualitative investigation of the communities detected by ComSim reveals that it proposes more balanced communities.

- **13.DCEIL: Distributed Community Detection with the CEIL Score (IEEE HPCC 2017)**
  - Akash Jain, Rupesh Nasre, Balaraman Ravindran 
  - [[Paper]](https://ieeexplore.ieee.org/document/8291922)
  - [[Java Reference]](https://github.com/RBC-DSAI-IITM/DCEIL)
  
Community detection in complex networks has a wide range of applications such as detection of cyber-communities in social networks, recommendations based on the interest group, and estimating hidden features in a social network. In distributed frameworks, the primary focus has been scalability. However, the accuracy of the algorithm's output is also critical. We propose the first distributed community detection algorithm based on the state-of-the-art CEIL scoring function. Our algorithm, named DCEIL, is fast, scalable and maintains the quality of communities. DCEIL outperforms the existing state-of-the-art distributed Louvain algorithm by 180% on an average in Normalized Mutual Information (NMI) Index and 6.61% on an average in Jaccard Index metrics. DCEIL completes execution for 1 billion edges within 112 minutes and outperforms state-of-the-art distributed Louvain algorithm by 4.3 ×. DCEIL critically exploits three novel heuristics which address the existing issues with distributed community detection algorithms that have the hierarchical structure of CEIL or Louvain methods. Further, our proposed heuristics are generic as well as efficient, and we illustrate their efficacy by enhancing the accuracy of distributed Louvain algorithm by 22.91% on an average in Jaccard Index, and the average execution time by 1.68 × over popular datasets.

- **14.A Community Detection Algorithm Using Network Topologies and Rule-Based Hierarchical Arc-Merging Strategies (PLOS One 2017)**
  - Yu-Hsiang Fu, Chung-Yuan Huang, and Chuen-Tsai Sun 
  - [[Paper]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0187603)
  - [[Python Reference]](https://github.com/yuhsiangfu/Hierarchical-Arc-Merging)

The authors use four criteria to examine a novel community detection algorithm: (a) **effectiveness** in terms of producing high values of normalized mutual information (NMI) and modularity, using well-known social networks for testing; (b) **examination**, meaning the ability to examine mitigating resolution limit problems using NMI values and synthetic networks; (c) **correctness**, meaning the ability to identify useful community structure results in terms of NMI values and Lancichinetti-Fortunato-Radicchi (LFR) benchmark networks; and (d) **scalability**, or the ability to produce comparable modularity values with fast execution times when working with large-scale real-world networks. In addition to describing a simple hierarchical arc-merging (HAM) algorithm that uses network topology information, we introduce rule-based arc-merging strategies for identifying community structures. Five well-studied social network datasets and eight sets of LFR benchmark networks were employed to validate the correctness of a ground-truth community, eight large-scale real-world complex networks were used to measure its efficiency, and two synthetic networks were used to determine its susceptibility to two resolution limit problems. Our experimental results indicate that the proposed HAM algorithm exhibited satisfactory performance efficiency, and that HAM-identified and ground-truth communities were comparable in terms of social and LFR benchmark networks, while mitigating resolution limit problems.

- **15.Local Higher-Order Graph Clustering (KDD 2017)**
  - Hao Yin, Austin Benson, Jure Leskovec, and David Gleich 
  - [[Paper]](https://cs.stanford.edu/~jure/pubs/mappr-kdd17.pdf)
  - [[Python Reference]](https://github.com/thecamelrider/MAPPR-Community-detection)
  - [[C++ SNAP Reference]](http://snap.stanford.edu/mappr/code.html)
  
Local graph clustering methods aim to nd a cluster of nodes by exploring a small region of the graph. ese methods are aractive
because they enable targeted clustering around a given seed node and are faster than traditional global graph clustering methods
because their runtime does not depend on the size of the input graph. However, current local graph partitioning methods are not
designed to account for the higher-order structures crucial to the network, nor can they eectively handle directed networks. Here
we introduce a new class of local graph clustering methods that address these issues by incorporating higher-order network information captured by small subgraphs, also called network motifs.We develop the Motif-based Approximate Personalized PageRank (MAPPR) algorithm that nds clusters containing a seed node with minimal motif conductance, a generalization of the conductance metric for network motifs. We generalize existing theory to prove the fast running time (independent of the size of the graph) and obtain theoretical guarantees on the cluster quality (in terms of motif conductance). We also develop a theory of node neighborhoods for nding sets that have small motif conductance, and apply these results to the case of nding good seed nodes to use as input to the MAPPR algorithm. Experimental validation on community detection tasks in both synthetic and real-world networks, shows that our new framework MAPPR outperforms the current edge-based personalized PageRank methodology.

- **16.ComSim: A Bipartite Community Detection Algorithm Using Cycle and Node’s Similarity (Complex Networks 2017)**
  - Raphael Tack, Fabien Tarissan, and Jean-Loup Guillaume
  - [[Paper]](https://arxiv.org/pdf/1705.04863.pdf)
  - [[C++ Reference]](https://github.com/rtackx/ComSim)
  
- **17.Discovering Community Structure in Multilayer Networks (DSAA 2017)**
  - Soumajit Pramanik, Raphael Tackx, Anchit Navelkar, Jean-Loup Guillaume, Bivas Mitra 
  - [[Paper]](https://ieeexplore.ieee.org/abstract/document/8259823)
  - [[Python Reference]](https://github.com/Soumajit-Pramanik/Multilayer-Louvain)
  
Community detection in single layer, isolated networks has been extensively studied in the past decade. However, many real-world systems can be naturally conceptualized as multilayer networks which embed multiple types of nodes and relations. In this paper, we propose algorithm for detecting communities in multilayer networks. The crux of the algorithm is based on the multilayer modularity index Q_M, developed in this paper. The proposed algorithm is parameter-free, scalable and adaptable to complex network structures. More importantly, it can simultaneously detect communities consisting of only single type, as well as multiple types of nodes (and edges). We develop a methodology to create synthetic networks with benchmark multilayer communities. We evaluate the performance of the proposed community detection algorithm both in the controlled environment (with synthetic benchmark communities) and on the empirical datasets (Yelp and Meetup datasets); in both cases, the proposed algorithm outperforms the competing state-of-the-art algorithms.

- **18.Evolutionary Graph Clustering for Protein Complex Identification (IEEE Transactions on Computational Biology and Bioinformatics  2016)**
  - Tiantian He and Keith C.C. Chan 
  - [[Paper]](https://ieeexplore.ieee.org/document/7792218)
  - [[Java Reference]](https://github.com/he-tiantian/EGCPI)

This paper presents a graph clustering algorithm, called EGCPI, to discover protein complexes in protein-protein interaction (PPI) networks. In performing its task, EGCPI takes into consideration both network topologies and attributes of interacting proteins, both of which have been shown to be important for protein complex discovery. EGCPI formulates the problem as an optimization problem and tackles it with evolutionary clustering. Given a PPI network, EGCPI first annotates each protein with corresponding attributes that are provided in Gene Ontology database. It then adopts a similarity measure to evaluate how similar the connected proteins are taking into consideration the network topology. Given this measure, EGCPI then discovers a number of graph clusters within which proteins are densely connected, based on an evolutionary strategy. At last, EGCPI identifies protein complexes in each discovered cluster based on the homogeneity of attributes performed by pairwise proteins. EGCPI has been tested with several real data sets and the experimental results show EGCPI is very effective on protein complex discovery, and the evolutionary clustering is helpful to identify protein complexes in PPI networks. The software of EGCPI can be downloaded via: https://github.com/ hetiantian1985/EGCPI.  

- **19.pSCAN: Fast and Exact Structural Graph Clustering (ICDE 2016)**
  - T Lijun Chang, Wei Li, Xuemin Lin, Lu Qin, and Wenjie Zhang 
  - [[Paper]](https://ieeexplore.ieee.org/document/7498245)
  - [[C++ Reference]](https://github.com/LijunChang/pSCAN)
  - [[Scala]](https://github.com/dawnranger/spark-pscan)

In this paper, we study the problem of structural graph clustering, a fundamental problem in managing and analyzing graph data. Given a large graph G = (V, E), structural graph clustering is to assign vertices in V to clusters and to identify the sets of hub vertices and outlier vertices as well, such that vertices in the same cluster are densely connected to each other while vertices in different clusters are loosely connected to each other. Firstly, we prove that the existing SCAN approach is worst-case optimal. Nevertheless, it is still not scalable to large graphs due to exhaustively computing structural similarity for every pair of adjacent vertices. Secondly, we make three observations about structural graph clustering, which present opportunities for further optimization. Based on these observations, in this paper we develop a new two-step paradigm for scalable structural graph clustering. Thirdly, following this paradigm, we present a new approach aiming to reduce the number of structural similarity computations. Moreover, we propose optimization techniques to speed up checking whether two vertices are structure-similar to each other. Finally, we conduct extensive performance studies on large real and synthetic graphs, which demonstrate that our new approach outperforms the state-of-the-art approaches by over one order of magnitude. Noticeably, for the twitter graph with 1 billion edges, our approach takes 25 minutes while the state-of-the-art approach cannot finish even after 24 hours.

- **20.Node-Centric Detection of Overlapping Communities in Social Networks (IWSCN 2016)**
  - Yehonatan Cohen, Danny Hendler, Amir Rubin
  - [[Paper]](https://arxiv.org/pdf/1607.01683.pdf)
  - [[Java Reference]](https://github.com/amirubin87/NECTAR)

We present NECTAR, a community detection algorithm that generalizes Louvain method’s local search heuristic for overlapping community structures. NECTAR chooses dynamically which objective function to optimize based on the network on which it is invoked. Our experimental evaluation on both synthetic benchmark graphs and real-world networks, based on ground-truth communities, shows that NECTAR provides excellent results as compared with state of the art community detection algorithms

- **21.Graph Clustering with Density-Cut (Arxiv 2016)**
  - Junming Shao, Qinli Yang, Jinhu Liu, Stefan Kramer
  - [[Paper]](https://arxiv.org/abs/1606.00950)
  - [[Go Reference]](https://github.com/askiada/GraphDensityCut)

How can we find a good graph clustering of a real-world network, that allows insight into its underlying structure and also potential functions? In this paper, we introduce a new graph clustering algorithm Dcut from a density point of view. The basic idea is to envision the graph clustering as a density-cut problem, such that the vertices in the same cluster are densely connected and the vertices between clusters are sparsely connected. To identify meaningful clusters (communities) in a graph, a density-connected tree is first constructed in a local fashion. Owing to the density-connected tree, Dcut allows partitioning a graph into multiple densely tight-knit clusters directly. We demonstrate that our method has several attractive benefits: (a) Dcut provides an intuitive criterion to evaluate the goodness of a graph clustering in a more natural and precise way; (b) Built upon the density-connected tree, Dcut allows identifying the meaningful graph clusters of densely connected vertices efficiently; (c) The density-connected tree provides a connectivity map of vertices in a graph from a local density perspective. We systematically evaluate our new clustering approach on synthetic as well as real data to demonstrate its good performance.

- **22.Community Detection in Directed Acyclic Graphs (European Physical Journal B 2015)**
  - Leo Speidel, Taro Takaguchi, Naoki Masuda
  - [[Paper]](https://link.springer.com/article/10.1140/epjb/e2015-60226-y)
  - [[Python Reference]](https://github.com/leospeidel/dag_community_paper)

Some temporal networks, most notably citation networks, are naturally represented as directed acyclic graphs (DAGs). To detect communities in DAGs, we propose a modularity for DAGs by defining an appropriate null model (i.e., randomized network) respecting the order of nodes. We implement a spectral method to approximately maximize the proposed modularity measure and test the method on citation networks and other DAGs. We find that the attained values of the modularity for DAGs are similar for partitions that we obtain by maximizing the proposed modularity (designed for DAGs), the modularity for undirected networks and that for general directed networks. In other words, if we neglect the order imposed on nodes (and the direction of links) in a given DAG and maximize the conventional modularity measure, the obtained partition is close to the optimal one in the sense of the modularity for DAGs.

- **23.Intra-Graph Clustering Using Collaborative Similarity Measure (DPCD 2015)**
  - Waqas Nawaz, Kifayat-Ullah Khan, Young-Koo Lee, and Sungyoung Lee
  - [[Paper]](https://link.springer.com/article/10.1007/s10619-014-7170-x)
  - [[Java Reference]](https://github.com/WNawaz/CSM)

Graph is an extremely versatile data structure in terms of its expressiveness and flexibility to model a range of real life phenomenon. Various networks like social networks, sensor networks and computer networks are represented and stored in the form of graphs. The analysis of these kind of graphs has an immense importance from quite a long time. It is performed from various aspects to get maximum out of such multifaceted information repository. When the analysis is targeted towards finding groups of vertices based on their similarity in a graph, clustering is the most conspicuous option. Previous graph clustering approaches either focus on the topological structure or attributes likeness, however, few recent methods constitutes both aspects simultaneously. Due to enormous computation requirements for similarity estimation, these methods are often suffered from scalability issues. In order to overcome this limitation, we introduce collaborative similarity measure (CSM) for intra-graph clustering. CSM is based on shortest path strategy, instead of all paths, to define structural and semantic relevance among vertices. First, we calculate the pair-wise similarity among vertices using CSM. Second, vertices are grouped together based on calculated similarity under k-Medoid framework. Empirical analysis, based on density, and entropy, proves the efficacy of CSM over existing measures. Moreover, CSM becomes a potential candidate for medium scaled graph analysis due to an order of magnitude less computations.

- **24.K-Clique Community Detection in Social NetworksBased on Formal Concept Analysis (IEEE Systems 2015)**
  -  Fei Hao, Geyong Min, Zheng Pei, Doo-Soon Park, Laurence T. Yang 
  - [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7117352&tag=1)
  - [[Python Reference]](https://github.com/indiansher/k-clique-community-detection-spark)

With the advent of ubiquitous sensing and networking, future social networks turn into cyber-physical interactions, which are attached with associated social attributes. Therefore, social network analysis is advancing the interconnections among cyber, physical, and social spaces. Community detection is an important issue in social network analysis. Users in a social network usually have some social interactions with their friends in a community because of their common interests or similar profiles. In this paper, an efficient algorithm of k-clique community detection using formal concept analysis (FCA) - a typical computational intelligence technique, namely, FCA-based k-clique community detection algorithm, is proposed. First, a formal context is constructed from a given social network by a modified adjacency matrix. Second, we define a type of special concept named k-equiconcept, which has the same k-size of extent and intent in a formal concept lattice. Then, we prove that the k-clique detection problem is equivalent to finding the k-equiconcepts. Finally, the efficient algorithms for detecting the k-cliques and k-clique communities are devised by virtue of k-equiconcepts and k-intent concepts, respectively. Experimental results demonstrate that the proposed algorithm has a higher F-measure value and significantly reduces the computational cost compared with previous works. In addition, a correlation between k and the number of k-clique communities is investigated.

- **25.High Quality, Scalable and Parallel Community Detection for Large Real Graphs (WWW 2014)**
  - Arnau Prat-Perez David Dominguez-Sal and Josep-Lluis Larriba-Pey
  - [[Paper]](http://wwwconference.org/proceedings/www2014/proceedings/p225.pdf)
  - [[C++ Reference]](https://github.com/Het-SCD/Het-SCD)

Community detection has arisen as one of the most relevant topics in the field of graph mining, principally for its applications in domains such as social or biological networks analysis. Different community detection algorithms have been proposed during the last decade, approaching the problem
from different perspectives. However, existing algorithms are, in general, based on complex and expensive computations, making them unsuitable for large graphs with millions of vertices and edges such as those usually found in the real world.
In this paper, we propose a novel disjoint community detection algorithm called Scalable Community Detection(SCD). By combining different strategies, SCD partitions the graph by maximizing the Weighted Community Clustering (W CC), a recently proposed community detection metric based on triangle analysis. Using real graphs with ground truth overlapped communities, we show that SCD outperforms the current state of the art proposals (even those aimed at finding overlapping communities) in terms of quality and performance. SCD provides the speed of the fastest algorithms and the quality in terms of NMI and F1Score of the most accurate state of the art proposals. We show that SCD is able to run up to two orders of magnitude faster than practical existing solutions by exploiting the parallelism of current multi-core processors, enabling us to process graphs of unprecedented size in short execution times.

- **26.GMAC: A Seed-Insensitive Approach to Local Community Detection (DaWak 2013)**
  - Lianhang Ma, Hao Huang, Qinming He, Kevin Chiew, Jianan Wu, and Yanzhe Che
  - [[Paper]](https://link.springer.com/chapter/10.1007/978-3-642-40131-2_26)
  - [[Python Reference]](https://github.com/SnehaManjunatha/Local-Community-Detection)

Local community detection aims at finding a community structure starting from a seed (i.e., a given vertex) in a network without global information, such as online social networks that are too large and dynamic to ever be known fully. Nonetheless, the existing approaches to local community detection are usually sensitive to seeds, i.e., some seeds may lead to missing of some true communities. In this paper, we present a seed-insensitive method called GMAC for local community detection. It estimates the similarity between vertices via the investigation on vertices’ neighborhoods, and reveals a local community by maximizing its internal similarity and minimizing its external similarity simultaneously. Extensive experimental results on both synthetic and real-world data sets verify the effectiveness of our GMAC algorithm.

- **27.On the Maximum Quasi-Clique Problem (Discrete Applied Mathematics 2013)**
  - Jeffrey Pattillo, Alexander Veremyev, Sergiy Butenko, and Vladimir Boginski
  - [[Paper]](https://www.sciencedirect.com/science/article/pii/S0166218X12002843)
  - [[Python Reference]](https://github.com/vishwass/QUAC)

Given a simple undirected graph  and a constant , a subset of vertices is called a -quasi-clique or, simply, a -clique if it induces a subgraph with the edge density of at least . The maximum -clique problem consists in finding a -clique of largest cardinality in the graph. Despite numerous practical applications, this problem has not been rigorously studied from the mathematical perspective, and no exact solution methods have been proposed in the literature. This paper, for the first time, establishes some fundamental properties of the maximum -clique problem, including the NP-completeness of its decision version for any fixed  satisfying , the quasi-heredity property, and analytical upper bounds on the size of a maximum -clique. Moreover, mathematical programming formulations of the problem are proposed and results of preliminary numerical experiments using a state-of-the-art optimization solver to find exact solutions are presented.

- **28.Community Detection in Networks with Node Attributes (ICDM 2013)**
  - Jaewon Yang, Julian McAuley, and Jure Leskovec
  - [[Paper]](https://www-cs.stanford.edu/~jure/pubs/cesna-icdm13.pdf)
  - [[C++ Reference]](https://github.com/snap-stanford/snap/tree/master/examples/cesna)
  
- **29.Detecting the Structure of Social Networks Using (α,β)-Communities (IWAMW 2011)**
  - Jing He, John Hopcroft, Liang Hongyu, Supasorn Suwajanakorn, and Liaoruo Wang
  - [[Paper]](https://ecommons.cornell.edu/bitstream/handle/1813/22415/WAW2011.pdf?sequence=2&isAllowed=y)
  - [[Python Reference]](https://github.com/handasontam/Alpha-Beta-Communities)
  
![image](https://raw.githubusercontent.com/JimWongM/ImageHost/master/img/20200406155249.png)

- **30.Multi-Netclust: An Efficient Tool for Finding Connected Clusters in Multi-Parametric Networks (IWAMW 2011)**
  - Arnold Kuzniar, Somdutta Dhir, Harm Nijveen, Sándor Pongor, Jack A.M. Leunissen 
  - [[Paper]](https://www.ncbi.nlm.nih.gov/pubmed/20679333)
  - [[C Reference]](https://github.com/arnikz/netclust)

Multi-netclust is a simple tool that allows users to extract connected clusters of data represented by different networks given in the form of matrices. The tool uses user-defined threshold values to combine the matrices, and uses a straightforward, memory-efficient graph algorithm to find clusters that are connected in all or in either of the networks. The tool is written in C/C++ and is available either as a form-based or as a command-line-based program running on Linux platforms. The algorithm is fast, processing a network of > 10(6) nodes and 10(8) edges takes only a few minutes on an ordinary computer.

- **31.Detecting Communities in Networks by Merging Cliques (IEEE ICICISYS 2009)**
  - Bowen Yan and Steve Gregory 
  - [[Paper]](https://ieeexplore.ieee.org/abstract/document/5358036)
  - [[Java Reference]](https://github.com/bowenyan/CommunityDetection-CliqueMod)

Many algorithms have been proposed for detecting disjoint communities (relatively densely connected subgraphs) in networks. One popular technique is to optimize modularity, a measure of the quality of a partition in terms of the number of intracommunity and intercommunity edges. Greedy approximate algorithms for maximizing modularity can be very fast and effective. We propose a new algorithm that starts by detecting disjoint cliques and then merges these to optimize modularity. We show that this performs better than other similar algorithms in terms of both modularity and execution speed.

- **32.Fast Unfolding of Communities in Large Networks (Journal of Statistical Mechanics 2008)**
  - Vincent D Blondel, Jean-Loup Guillaume, Renaud Lambiotte, Renaud Lefebvre
  - [[Paper]](https://arxiv.org/abs/0803.0476)
  - [[Python]](https://python-louvain.readthedocs.io/en/latest/)
  - [[Python]](https://github.com/tsakim/Shuffled_Louvain)
  - [[Python]](https://github.com/brady131313/Louvains)
  - [[C]](https://github.com/jlguillaume/louvain)
  - [[C++ Parallel]](https://github.com/Exa-Graph/vite)
  - [[C++]](https://github.com/Sotera/spark-distributed-louvain-modularity)
  - [[C++]](https://github.com/ghp-16/louvain)
  - [[C++ GPU]](https://github.com/bobbys-dev/gpu-louvain)
  - [[Javascript]](https://github.com/upphiminn/jLouvain)
  - [[Javascript]](https://github.com/multivacplatform/louvain)
  - [[Javascript]](https://github.com/graphology/graphology-communities-louvain)
  - [[Java]](https://github.com/JoanWu5/Louvain)
  - [[Matlab]](https://github.com/JoanWu5/Louvain)
  - [[Matlab]](https://github.com/BigChopH/Louvain-clustering)
  - [[Scala]](https://github.com/Sotera/spark-distributed-louvain-modularity)
  - [[Rust]](https://github.com/graphext/louvain-rs)
  
- **33.Modularity and Community Detection in Bipartite Networks (Phys. Rev. E 2007)**
  - Michael J. Barberl
  - [[Paper]](https://arxiv.org/abs/0707.1616)
  - [[Python Reference]](https://github.com/genisott/pycondor)

The modularity of a network quantifies the extent, relative to a null model network, to which vertices cluster into community groups. We define a null model appropriate for bipartite networks, and use it to define a bipartite modularity. The bipartite modularity is presented in terms of a modularity matrix B; some key properties of the eigenspectrum of B are identified and used to describe an algorithm for identifying modules in bipartite networks. The algorithm is based on the idea that the modules in the two parts of the network are dependent, with each part mutually being used to induce the vertices for the other part into the modules. We apply the algorithm to real-world network data, showing that the algorithm successfully identifies the modular structure of bipartite networks.
  
