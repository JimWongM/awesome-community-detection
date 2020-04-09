
## Physics Inspired

- **1.Community Detection Using Preference Networks (Physica A 2018)**
  - Mursel Tasgin and Halu Bingol
  - [[Paper]](https://arxiv.org/pdf/1708.08305.pdf)
  - [[Java Reference]](https://github.com/murselTasginBoun/CDPN)

Community detection is the task of identifying clusters or groups of nodes in a network where
nodes within the same group are more connected with each other than with nodes in different groups.
It has practical uses in identifying similar functions or roles of nodes in many biological, social and
computer networks. With the availability of very large networks in recent years, performance and
scalability of community detection algorithms become crucial, i.e. if time complexity of an algorithm
is high, it can not run on large networks. In this paper, we propose a new community detection
algorithm, which has a local approach and is able to run on large networks. It has a simple and
effective method; given a network, algorithm constructs a preference network of nodes where each
node has a single outgoing edge showing its preferred node to be in the same community with. In such
a preference network, each connected component is a community. Selection of the preferred node is
performed using similarity based metrics of nodes. We use two alternatives for this purpose which
can be calculated in 1-neighborhood of nodes, i.e. number of common neighbors of selector node and
its neighbors and, the spread capability of neighbors around the selector node which is calculated
by the gossip algorithm of Lind et.al. Our algorithm is tested on both computer generated LFR
networks and real-life networks with ground-truth community structure. It can identify communities
accurately in a fast way. It is local, scalable and suitable for distributed execution on large networks.

- **2.Thermodynamics of the Minimum Description Length on Community Detection (ArXiv 2018)**
  - Juan Ignacio Perotti, Claudio Juan Tessone, Aaron Clauset and Guido Caldarelli
  - [[Paper]](https://arxiv.org/pdf/1806.07005.pdf)
  - [[Python Reference]](https://github.com/jipphysics/bmdl_edm)

Modern statistical modeling is an important complement to the more traditional approach of physics where Complex Systems are studied by means of extremely simple idealized models. The Minimum Description Length (MDL) is a principled approach to statistical modeling combining
Occam’s razor with Information Theory for the selection of models providing the most concise descriptions. In this work, we introduce the Boltzmannian MDL (BMDL), a formalization of the principle of MDL with a parametric complexity conveniently formulated as the free-energy of an
artificial thermodynamic system. In this way, we leverage on the rich theoretical and technical background of statistical mechanics, to show the crucial importance that phase transitions and other thermodynamic concepts have on the problem of statistical modeling from an information
theoretic point of view. For example, we provide information theoretic justifications of why a hightemperature series expansion can be used to compute systematic approximations of the BMDL when the formalism is used to model data, and why statistically significant model selections can be
identified with ordered phases when the BMDL is used to model models. To test the introduced formalism, we compute approximations of BMDL for the problem of community detection in complex networks, where we obtain a principled MDL derivation of the Girvan-Newman (GN) modularity and
the Zhang-Moore (ZM) community detection method. Here, by means of analytical estimations and numerical experiments on synthetic and empirical networks, we find that BMDL-based correction terms of the GN modularity improve the quality of the detected communities and we also find an
information theoretic justification of why the ZM criterion for estimation of the number of network communities is better than alternative approaches such as the bare minimization of a free energy. Finally, we discuss several research questions for future works, contemplating the general nature of the BMDL and its application to the particular problem of community detection in complex networks.


- **3.Fluid Communities: A Community Detection Algorithm (CompleNet 2017)**
  - Ferran Parés, Dario Garcia-Gasulla, Armand Vilalta, Jonatan Moreno, Eduard Ayguadé, Jesús Labarta, Ulises Cortés and Toyotaro Suzumura
  - [[Paper]](https://arxiv.org/abs/1703.09307)
  - [[Python Reference]](https://github.com/HPAI-BSC/Fluid-Communities)

We introduce a community detection algorithm (Fluid Communities) based on the idea of fluids interacting in an environment, expanding and contracting as a result of that interaction. Fluid Communities is based on the propagation methodology, which represents the state-of-the-art in terms of computational cost and scalability. While being highly efficient, Fluid Communities is able to find communities in synthetic graphs with an accuracy close to the current best alternatives. Additionally, Fluid Communities is the first propagation-based algorithm capable of identifying a variable number of communities in network. To illustrate the relevance of the algorithm, we evaluate the diversity of the communities found by Fluid Communities, and find them to be significantly different from the ones found by alternative methods.

- **4.A Local Perspective on Community Structure in Multilayer Networks (Network Science 2017)**
  - Lucas GS Jeub, Michael Mahoney, Peter J Mucha and Mason A Porter
  - [[Paper]](https://arxiv.org/pdf/1510.05185.pdf)
  - [[Python Reference]](https://github.com/LJeub/LocalCommunities)
  
- **5.BlackHole: Robust Community Detection Inspired by Graph Drawing (ICDE 2016)**
  -  Sungsu Lim, Junghoon Kim, and Jae-Gil Lee 
  - [[Paper]](https://ieeexplore.ieee.org/document/7498226)
  - [[C++ Reference]](https://github.com/thousfeet/Blackhole-Community-detection)

With regard to social network analysis, we concentrate on two widely-accepted building blocks: community detection and graph drawing. Although community detection and graph drawing have been studied separately, they have a great commonality, which means that it is possible to advance one field using the techniques of the other. In this paper, we propose a novel community detection algorithm for undirected graphs, called BlackHole, by importing a geometric embedding technique from graph drawing. Our proposed algorithm **transforms the vertices of a graph to a set of points on a low-dimensional space** whose coordinates are determined by a variant of graph drawing algorithms, following the overall procedure of spectral clustering. The set of points are then clustered using a conventional clustering algorithm to form communities. Our primary contribution is to prove that a common idea in graph drawing, which is characterized by consideration of repulsive forces in addition to attractive forces, improves the clusterability of an embedding. As a result, our algorithm has the advantages of being robust especially when the community structure is not easily detectable. Through extensive experiments, we have shown that BlackHole achieves the accuracy higher than or comparable to the state-of-the-art algorithms.

- **6.Defining Least Community as a Homogeneous Group in Complex Networks (Physica A 2015)**
  - Renaud Lambiotte, J-C Delvenne, and Mauricio Barahona
  - [[Paper]](https://arxiv.org/pdf/1502.00284.pdf)
  - [[Python Reference]](https://github.com/dingmartin/HeadTailCommunityDetection)

This paper introduces a new concept of least community that is as homogeneous as a random graph,
and develops a new community detection algorithm from the perspective of homogeneity or
heterogeneity. Based on this concept, we adopt head/tail breaks – a newly developed classification
scheme for data with a heavy-tailed distribution – and rely on edge betweenness given its heavy-tailed
distribution to iteratively partition a network into many heterogeneous and homogeneous communities.
Surprisingly, the derived communities for any self-organized and/or self-evolved large networks
demonstrate very striking power laws, implying that there are far more small communities than large
ones. This notion of far more small things than large ones constitutes a new fundamental way of
thinking for community detection. 


- **7.Community Detection Based on Distance Dynamics (KDD 2015)**
  - Shao Junming, Han Zhichao, Yang Qinli, and Zhou Tao
  - [[Paper]](https://dl.acm.org/citation.cfm?id=2783301)
  - [[Python Reference]](https://github.com/chocolates/Community-detection-based-on-distance-dynamics)

In this paper, we introduce a new community detection algorithm, called Attractor, which automatically spots communities in a network by examining the changes of "distances" among nodes (i.e. distance dynamics). The fundamental idea is to envision the target network as an adaptive dynamical system, where each node interacts with its neighbors. The interaction will change the distances among nodes, while the distances will affect the interactions. Such interplay eventually leads to a steady distribution of distances, where the nodes sharing the same community move together and the nodes in different communities keep far away from each other. Building upon the distance dynamics, Attractor has several remarkable advantages: (a) It provides an intuitive way to analyze the community structure of a network, and more importantly, faithfully captures the natural communities (with high quality). (b) Attractor allows detecting communities on large-scale networks due to its low time complexity (O(|E|)). (c) Attractor is capable of discovering communities of arbitrary size, and thus small-size communities or anomalies, usually existing in real-world networks, can be well pinpointed. Extensive experiments show that our algorithm allows the effective and efficient community detection and has good performance compared to state-of-the-art algorithms.

- **8.Think Locally, Act Locally: Detection of Small, Medium-Sized, and Large Communities in Large Networks (Physica Review E 2015)**
  - Lucas G. S. Jeub, Prakash Balachandran, Mason A. Porter, Peter J. Mucha, and Michael W. Mahoney
  - [[Paper]](https://arxiv.org/abs/1403.3795v1)
  - [[Python Reference]](https://github.com/LJeub/LocalCommunities)

It is common in the study of networks to investigate meso-scale features to try to understand network structure and function. For example, numerous algorithms have been developed to try to identify ``communities,'' which are typically construed as sets of nodes with denser connections internally than with the remainder of a network. In this paper, we adopt a complementary perspective that ``communities'' are associated with bottlenecks of dynamical processes that begin at locally-biased seed sets of nodes, and we employ several different community-identification procedures to investigate community quality as a function of community size. Using several empirical and synthetic networks, we identify several distinct scenarios for ``size-resolved community structure'' that can arise in real (and realistic) networks: (i) the best small groups of nodes can be better than the best large groups (for a given formulation of the idea of a good community); (ii) the best small groups can have a quality that is comparable to the best medium-sized and large groups; and (iii) the best small groups of nodes can be worse than the best large groups. As we discuss in detail, which of these three cases holds for a given network can make an enormous difference when investigating and making claims about network community structure, and it is important to take this into account to obtain reliable downstream conclusions.

- **9.Detecting Community Structure Using Label Propagation with Weighted Coherent Neighborhood Propinquity (Physica A 2013)**
  - Hao Lou, Shenghong Li, and Yuxin Zhao
  - [[Paper]](https://www.sciencedirect.com/science/article/pii/S0378437113002173)
  - [[Java Reference]](https://github.com/mahdiabdollahpour/Detecting-Community-Structure/tree/master/src)

Community detection has become an important methodology to understand the organization and function of various real-world networks. The label propagation algorithm (LPA) is an almost linear time algorithm proved to be effective in finding a good community structure. However, LPA has a limitation caused by its one-hop horizon. Specifically, each node in LPA adopts the label shared by most of its one-hop neighbors; much network topology information is lost in this process, which we believe is one of the main reasons for its instability and poor performance. Therefore in this paper we introduce a measure named weighted coherent neighborhood propinquity (weighted-CNP) to represent the probability that a pair of vertices are involved in the same community. In label update, a node adopts the label that has the maximum weighted-CNP instead of the one that is shared by most of its neighbors. We propose a dynamic and adaptive weighted-CNP called entropic-CNP by using the principal of entropy to modulate the weights. Furthermore, we propose a framework to integrate the weighted-CNP in other algorithms in detecting community structure. We test our algorithm on both computer-generated networks and real-world networks. The experimental results show that our algorithm is more robust and effective than LPA in large-scale networks.

- **10.Parallel Community Detection on Large Networks with Propinquity Dynamics (KDD 2009)**
  - Yuzhou Zhang, Jianyong Wang, Yi Wang, and Lizhu Zhou 
  - [[Paper]](https://grid.cs.gsu.edu/~myan2/communitydetection/16.pdf)
  - [[Java Reference]](https://github.com/csdashes/GraphStreamCommunityDetection)
    
- **11.Laplacian Dynamics and Multiscale Modular Structure in Networks (IEEE TNSE 2008)**
  - Renaud Lambiotte, J-C Delvenne, and Mauricio Barahona
  - [[Paper]](https://arxiv.org/abs/0812.1770v3)
  - [[R Reference]](https://github.com/buzzlumberjack/Communities-Detection)
  
- **12.Statistical Mechanics of Community Detection (Phyics Review E 2006)**
  - Jorh Reichardt and Stefan Bornholdt
  - [[Paper]](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.74.016110)
  - [[Ruby Reference]](https://github.com/duett/community-detection)
