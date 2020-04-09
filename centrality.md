## Centrality and Cuts

- **1.A Novel Graph-based Clustering Method Using Noise Cutting (Information Systems 2020)**
  - Lin-Tao Li, Zhong-Yang Xiong, Qi-Zhu Dai, Yong-Fang Zha. Yu-Fang Zhang, Jing-Pei Dan
  - [[Paper]](https://www.sciencedirect.com/science/article/pii/S0306437920300156)
  - [[Matlab Reference]](https://github.com/lintao6/CutPC)


Recently, many methods have appeared in the field of cluster analysis. Most existing clustering algorithms have considerable limitations in dealing with local and nonlinear data patterns. Algorithms based on graphs provide good results for this problem. However, some widely used graph-based clustering methods, such as spectral clustering algorithms, are sensitive to noise and outliers. In this paper, a cut-point clustering algorithm (CutPC) based on a natural neighbor graph is proposed. The CutPC method performs noise cutting when a cut-point value is above the critical value. Normally, the method can automatically identify clusters with arbitrary shapes and detect outliers **without any prior knowledge or preparatory parameter settings**. The user can also adjust a coefficient to adapt clustering solutions for particular problems better. Experimental results on various synthetic and real-world datasets demonstrate the obvious superiority of CutPC compared with k-means, DBSCAN, DPC, SC, and DCore.

- **2.Hypergraph Clustering with Categorical Edge Labels (Arxiv 2019)**
  - Ilya Amburg, Nate Veldt, Austin R. Benson
  - [[Paper]](https://arxiv.org/pdf/1910.09943.pdf)
  - [[Julia Reference]](https://github.com/nveldt/CategoricalEdgeClustering)

Graphs and networks are a standard model for describing data or systems based on pairwise interactions. Oftentimes, the underlying relationships involve more than two entities at a time, and hypergraphs are a more faithful model. However, we have fewer rigorous methods that can provide insight from such representations. Here, we develop a computational framework for the problem of clustering hypergraphs with categorical edge labels --- or different interaction types --- where clusters corresponds to groups of nodes that frequently participate in the same type of interaction.

Our methodology is based on a combinatorial objective function that is related to correlation clustering but enables the design of much more efficient algorithms. When there are only two label types, our objective can be optimized in polynomial time, using an algorithm based on minimum cuts. Minimizing our objective becomes NP-hard with more than two label types, but we develop fast approximation algorithms based on linear programming relaxations that have theoretical cluster quality guarantees. We demonstrate the efficacy of our algorithms and the scope of the model through problems in edge-label community detection, clustering with temporal data, and exploratory data analysis.


- **3.Learning Resolution Parameters for Graph Clustering (WWW 2019)**
  - Nate Veldt, David F. Gleich, Anthony Wirth
  - [[Paper]](https://arxiv.org/abs/1903.05246)
  - [[Julia Reference]](https://github.com/nveldt/LearnResParams)

Finding clusters of well-connected nodes in a graph is an extensively studied problem in graph-based data analysis. Because of its many applications, a large number of distinct graph clustering objective functions and algorithms have already been proposed and analyzed. To aid practitioners in determining the best clustering approach to use in different applications, we present new techniques for automatically learning how to set clustering resolution parameters. These parameters control the size and structure of communities that are formed by optimizing a generalized objective function. We begin by formalizing the notion of a parameter fitness function, which measures how well a fixed input clustering approximately solves a generalized clustering objective for a specific resolution parameter value. Under reasonable assumptions, which suit two key graph clustering applications, such a parameter fitness function can be efficiently minimized using a bisection-like method, yielding a resolution parameter that fits well with the example clustering. We view our framework as a type of single-shot hyperparameter tuning, as we are able to learn a good resolution parameter with just a single example. Our general approach can be applied to learn resolution parameters for both local and global graph clustering objectives. We demonstrate its utility in several experiments on real-world data where it is helpful to learn resolution parameters from a given example clustering.


- **4.Parallelizing Pruning-based Graph Structural Clustering (ICPP 2018)**
  - Yulin Che, Shixuan Sun, and Qiong Luo
  - [[Paper]](https://dl.acm.org/citation.cfm?doid=3225058.3225063)
  - [[C++ Reference]](https://github.com/GraphProcessor/ppSCAN)
  
- **5.Real-Time Community Detection in Large Social Networks on a Laptop (PLOS 2018)**
  - Benjamin Paul Chamberlain, Josh Levy-Kramer, Clive Humby, and Marc Peter Deisenroth
  - [[Paper]](https://arxiv.org/pdf/1601.03958.pdf)
  - [[Python Reference]](https://github.com/melifluos/LSH-community-detection)
  
- **6.A Polynomial Algorithm for Balanced Clustering via Graph Partitioning (EJOR 2018)**
  - Luis-Evaristo Caraballo, José-Miguel Díaz-Báñez, Nadine Kroher
  - [[Paper]](https://arxiv.org/abs/1801.03347)
  - [[Python Reference]](https://github.com/varocaraballo/graph_partition_clustering)
  
- **7.A Community Detection Algorithm Using Network Topologies and Rule-based Hierarchical Arc-merging Strategies (PLOS 2018)**
  - Yu-Hsiang Fu, Chung-Yuan Huang, and Chuen-Tsai Sun
  - [[Paper]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0187603)
  - [[Python Reference]](https://github.com/yuhsiangfu/Hierarchical-Arc-Merging)
  
- **Hidden Community Detection in Social Networks (Information Sciences 2018)**
  - Kun He, Yingru Li, Sucheta Soundarajan, John E. Hopcroft
  - [[Paper]](https://arxiv.org/pdf/1702.07462v1.pdf)
  - [[Python Reference]](https://github.com/JHL-HUST/HiCode)
  
- **8.Ego-splitting Framework: from Non-Overlapping to Overlapping Clusters (KDD 2017)**
  -  Alessandro Epasto, Silvio Lattanzi, and Renato Paes Leme
  - [[Paper]](https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p145.pdf)
  - [[Python Reference]](https://github.com/benedekrozemberczki/EgoSplitting)
  
- **9.Query-oriented Graph Clustering (PAKDD 2017)**
  -  Li-Yen Kuo, Chung-Kuang Chou, and Ming-Syan Chen
  - [[Paper]](https://link.springer.com/chapter/10.1007/978-3-319-57529-2_58)
  - [[Python Reference]](https://github.com/iankuoli/QGC)
  
- **10.Fast Heuristic Algorithm for Multi-scale Hierarchical Community Detection (ASONAM 2017)**
  - Eduar Castrillo, Elizabeth León, and Jonatan Gómez
  - [[Paper]](https://dl.acm.org/citation.cfm?doid=3110025.3110125)
  - [[C++ Reference]](https://github.com/eduarc/WMW)
  
- **11.Community Detection in Signed Networks: the Role of Negative Ties in Different Scales (Scientific Reports 2015)**
  - Pouya Esmailian and Mahdi Jalili
  - [[Paper]](https://www.nature.com/articles/srep14339)
  - [[Java Reference]](https://github.com/pouyaesm/signed-community-detection)  
  
- **12.Detecting Community Structures in Social Networks by Graph Sparsification (CODS 2016)**
  - Partha Basuchowdhuri, Satyaki Sikdar, Sonu Shreshtha, and Subhasis Majumder
  - [[Paper]](http://dl.acm.org/citation.cfm?id=2888479)
  - [[Python Reference]](https://github.com/satyakisikdar/spanner-comm-detection)
  
- **13.Community Detection in Complex Networks Using Density-Based Clustering Algorithm and Manifold Learning (Physica A 2016)**
  - Tao Youa, Hui-Min Chenga, Yi-Zi Ninga, Ben-Chang Shiab, and Zhong-Yuan Zhang
  - [[Paper]](https://www.sciencedirect.com/science/article/pii/S0378437116304563)
  - [[Matlab Reference]](https://github.com/isaac-you/IsoFpd)
  
- **14.Smart Partitioning of Geo-Distributed Resources to Improve Cloud Network Performance (CloudNet 2015)**
  - Hooman Peiro Sajjad, Fatemeh Rahimian, and Vladimir Vlassov 
  - [[Paper]](https://ieeexplore.ieee.org/document/7335292)
  - [[Java Reference]](https://github.com/shps/mdc-community-detection)
  
- **15.Generalized Modularity for Community Detection (ECML 2015)**
  - Mohadeseh Ganji, Abbas Seifi, Hosein Alizadeh, James Bailey, and Peter J. Stuckey
  - [[Paper]](https://people.eng.unimelb.edu.au/mganji/papers/ECML15.pdf)
  - [[Python Reference]](https://github.com/xiaoylu/ResolutionCommunityDetection)
  
- **16.General Optimization Technique for High-quality Community Detection in Complex Networks (Physical Review E 2014)**
  - Stanislav Sobolevsky, Riccardo Campari, Alexander Belyi, and Carlo Ratti
  - [[Paper]](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.90.012811)
  - [[Python Reference]](https://github.com/Casyfill/pyCombo)
  
- **17.Online Community Detection for Large Complex Networks (IJCAI 2013)**
  - Wangsheng Zhang, Gang Pan, Zhaohui Wu and Shijian Li
  - [[Paper]](https://www.ijcai.org/Proceedings/13/Papers/281.pdf)
  - [[C++ Reference]](https://github.com/isaac-you/IsoFpd)
  
- **18.Agglomerative Clustering via Maximum Incremental Path Integral (Pattern Recognition 2013)**
  - Wei Zhang, Deli Zhao, and Xiaogang Wang
  - [[Paper]](https://www.sciencedirect.com/science/article/pii/S0031320313001830)
  - [[Matlab Reference]](https://github.com/waynezhanghk/gactoolbox)
  
- **19.Graph Degree Linkage: Agglomerative Clustering on a Directed Graph (ECCV 2012)**
  - Wei Zhang, Xiaogang Wang, Deli Zhao and Xiaoou Tang
  - [[Paper]](https://arxiv.org/abs/1208.5092)
  - [[Matlab Reference]](https://github.com/waynezhanghk/gactoolbox)
  - [[Python Reference]](https://github.com/myungjoon/GDL)
  
- **20.Weighted Graph Cuts without Eigenvectors a Multilevel Approach (IEEE TPAMI 2007)**
  - Inderjit S Dhillon, Brian Kulis, and Yuqiang Guan 
  - [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4302760)
  - [[C Reference]](https://github.com/iromu/Graclus)
