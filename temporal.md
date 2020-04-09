## Temporal Methods

- **1.Detecting Stable Communities in Link Streams at Multiple Temporal Scales (IEEE Big Data 2020)**
  -  Hongchao Qin, Rong-Hua Li, Guoren Wang, Xin Huang, Ye Yuan, and Jeffrey Xu Yu 
  - [[Paper]](https://ieeexplore.ieee.org/abstract/document/9001192)
  - [[Python Reference]](https://github.com/VeryLargeGraph/TSCAN)

Community detection is a fundamental task in graph data mining. Most existing studies in contact networks, collaboration networks, and social networks do not utilize the temporal information associated with edges for community detection. In this paper, we study a problem of finding stable communities in a temporal network, where each edge is associated with a timestamp. Our goal is to identify the communities in a temporal network that are stable over time. To efficiently find the stable communities, we develop a new community detection algorithm based on the density-based graph clustering framework. We also propose several carefully-designed pruning techniques to significantly speed up the proposed algorithm. We conduct extensive experiments on four real-life temporal networks to evaluate our algorithm. The results demonstrate the effectiveness and efficiency of the proposed algorithm.

- **2.Detecting Stable Communities in Link Streams at Multiple Temporal Scales (ECML-PKDD 2019)**
  - Souaad Boudebza, Remy Cazabet, Omar Nouali, and Faical Azouaou
  - [[Paper]](https://arxiv.org/pdf/1907.10453.pdf)
  - [[Python Reference]](https://github.com/VeryLargeGraph/TSCAN)

Link streams model interactions over time in a wide range of
fields. Under this model, the challenge is to mine efficiently both temporal and topological structures. Community detection and change point
detection are one of the most powerful tools to analyze such evolving
interactions. In this paper, we build on both to detect stable community
structures by identifying change points within meaningful communities.
Unlike existing dynamic community detection algorithms, the proposed
method is able to discover stable communities efficiently at multiple temporal scales. We test the effectiveness of our method on synthetic networks, and on high-resolution time-varying networks of contacts drawn
from real social networks.

- **3.DynComm R Package - Dynamic Community Detection for Evolving Networks (Arxiv 2019)**
  - Rui Portocarrero Sarmento, Luís Lemos, Mário Cordeiro, Giulio Rossetti, and Douglas Cardoso
  - [[Paper]](https://arxiv.org/abs/1905.01498)
  - [[R Reference]](https://github.com/softskillsgroup/DynComm-R-package)

Nowadays, the analysis of dynamics in networks represents a great deal in the Social Network Analysis research area. To support students, teachers, developers, and researchers in this work we introduce a novel R package, namely DynComm. It is designed to be a multi-language package, that can be used for community detection and analysis on dynamic networks. The package introduces interfaces to facilitate further developments and the addition of new and future developed algorithms to deal with community detection in evolving networks. This new package has the goal of abstracting the programmatic interface of the algorithms, whether they are written in R or other languages, and expose them as functions in R.


- **4.Block-Structure Based Time-Series Models For Graph Sequences (Arxiv 2018)**
  - Mehrnaz Amjadi and Theja Tulabandhula
  - [[Paper]](https://arxiv.org/pdf/1804.08796v2.pdf)
  - [[Python Reference]](https://github.com/thejat/dynamic-network-growth-models)

Although the computational and statistical trade-off for modeling single graphs,
for instance, using block models is relatively well understood, extending such results
to sequences of graphs has proven to be difficult. In this work, we take a step in
this direction by proposing two models for graph sequences that capture: (a) link
persistence between nodes across time, and (b) community persistence of each node
across time. In the first model, we assume that the latent community of each node does
not change over time, and in the second model we relax this assumption suitably. For
both of these proposed models, we provide statistically and computationally efficient
inference algorithms, whose unique feature is that they leverage community detection
methods that work on single graphs. We also provide experimental results validating
the suitability of our models and methods on synthetic and real instances.

- **5.DyPerm: Maximizing Permanence for Dynamic Community Detection (PKDD 2018)**
  - Prerna Agarwal, Richa Verma, Ayush Agarwal, Tanmoy Chakraborty
  - [[Paper]](https://arxiv.org/abs/1802.04593)
  - [[Python Reference]](https://github.com/ayush14029/Dyperm-Code)

In this paper, we propose DyPerm, the first dynamic community detection method which optimizes a novel community scoring metric, called permanence. DyPerm incrementally modifies the community structure by updating those communities where the editing of nodes and edges has been performed, keeping the rest of the network unchanged. We present strong theoretical guarantees to show how/why mere updates on the existing community structure leads to permanence maximization in dynamic networks, which in turn decreases the computational complexity drastically. Experiments on both synthetic and six real-world networks with given ground-truth community structure show that DyPerm achieves (on average) 35% gain in accuracy (based on NMI) compared to the best method among four baseline methods. DyPerm also turns out to be 15 times faster than its static counterpart.

-**6.OLCPM: An Online Framework for Detecting Overlapping Communities in Dynamic Social Networks (Computer Communciations, 2018)**
  - Souaad Boudebzaa, Remy Cazabet, Faical Azouaoua, Omar Nouali
  - [[Paper]](https://arxiv.org/pdf/1804.03842.pdf)
  - [[Java Reference]](http://olcpm.sci-web.net)

Community structure is one of the most prominent features of complex networks. Community structure detection is of great importance to provide insights
into the network structure and functionalities. Most proposals focus on static
networks. However, finding communities in a dynamic network is even more
challenging, especially when communities overlap with each other. In this article, we present an online algorithm, called OLCPM, based on clique percolation
and label propagation methods. OLCPM can detect overlapping communities
and works on temporal networks with a fine granularity. By locally updating
the community structure, OLCPM delivers significant improvement in running
time compared with previous clique percolation techniques. The experimental
results on both synthetic and real-world networks illustrate the effectiveness of
the method.


- **7.Temporally Evolving Community Detection and Prediction in Content-Centric Networks (ECML 2018)**
  - Ana Paula Appel, Renato L. F. Cunha, Charu C. Aggarwal, and Marcela Megumi Terakado 
  - [[Paper]](https://arxiv.org/pdf/1807.06560v1.pdf)
  - [[Python Reference]](https://github.com/renatolfc/chimera-stf)

Abstract—In this work, we consider the problem of combining
link, content and temporal analysis for community detection and
prediction in evolving networks. Such temporal and content-rich
networks occur in many real-life settings, such as bibliographic
networks and question answering forums. Most of the work in
the literature (that uses both content and structure) deals with
static snapshots of networks, and they do not reflect the dynamic changes occurring over multiple snapshots. Incorporating
dynamic changes in the communities into the analysis can also
provide useful insights about the changes in the network such
as the migration of authors across communities. In this work,
we propose Chimera1
, a shared factorization model that can
simultaneously account for graph links, content, and temporal
analysis. This approach works by extracting the latent semantic
structure of the network in multidimensional form, but in a
way that takes into account the temporal continuity of these
embeddings. Such an approach simplifies temporal analysis of
the underlying network by using the embedding as a surrogate.
A consequence of this simplification is that it is also possible
to use this temporal sequence of embeddings to predict future
communities. We present experimental results illustrating the
effectiveness of the approach.


- **8.A Streaming Algorithm for Graph Clustering (Arxiv 2017)**
  - Alexandre Hollocou, Julien Maudet, Thomas Bonald and Marc Lelarge
  - [[Paper]](https://arxiv.org/pdf/1712.04337v1.pdf)
  - [[C++ Reference]](https://github.com/ahollocou/graph-streaming)

We introduce a novel algorithm to perform graph clustering in the edge streaming
setting. In this model, the graph is presented as a sequence of edges that can be
processed strictly once. Our streaming algorithm has an extremely low memory
footprint as it stores only three integers per node and does not keep any edge in
memory. We provide a theoretical justification of the design of the algorithm based
on the modularity function, which is a usual metric to evaluate the quality of a
graph partition. We perform experiments on massive real-life graphs ranging from
one million to more than one billion edges and we show that this new algorithm
runs more than ten times faster than existing algorithms and leads to similar or
better detection scores on the largest graphs.

- **9.DynaMo: Dynamic Community Detection by Incrementally Maximizing Modularity (Arxiv 2017)**
  - Di Zhuang, J. Morris Chang, Mingchen Li
  - [[Paper]](https://arxiv.org/abs/1709.08350)
  - [[Java Reference]](https://github.com/nogrady/dynamo)

Community detection is of great importance for online social network analysis. The volume, variety and velocity of data generated by today's online social networks are advancing the way researchers analyze those networks. For instance, real-world networks, such as Facebook, LinkedIn and Twitter, are inherently growing rapidly and expanding aggressively over time. However, most of the studies so far have been focusing on detecting communities on the static networks. It is computationally expensive to directly employ a well-studied static algorithm repeatedly on the network snapshots of the dynamic networks. We propose DynaMo, a novel modularity-based dynamic community detection algorithm, aiming to detect communities of dynamic networks as effective as repeatedly applying static algorithms but in a more efficient way. DynaMo is an adaptive and incremental algorithm, which is designed for incrementally maximizing the modularity gain while updating the community structure of dynamic networks. In the experimental evaluation, a comprehensive comparison has been made among DynaMo, Louvain (static) and 5 other dynamic algorithms. Extensive experiments have been conducted on 6 real-world networks and 10,000 synthetic networks. Our results show that DynaMo outperforms all the other 5 dynamic algorithms in terms of the effectiveness, and is 2 to 5 times (by average) faster than Louvain algorithm.

- **10.Model-Based Clustering of Time-Evolving Networks through Temporal Exponential-Family Random Graph Models (Arxiv 2017)**
  - Kevin H. Lee, Lingzhou Xue, and David R. Hunter
  - [[Paper]](https://arxiv.org/abs/1712.07325)
  - [[R Reference]](https://github.com/amalag-19/dynERGM_R)

Dynamic networks are a general language for describing time-evolving complex systems, and discrete time network models provide an emerging statistical technique for various applications. It is a fundamental research question to detect the community structure in time-evolving networks. However, due to significant computational challenges and difficulties in modeling communities of time-evolving networks, there is little progress in the current literature to effectively find communities in time-evolving networks. In this work, we propose a novel model-based clustering framework for time-evolving networks based on discrete time exponential-family random graph models. To choose the number of communities, we use conditional likelihood to construct an effective model selection criterion. Furthermore, we propose an efficient variational expectation-maximization (EM) algorithm to find approximate maximum likelihood estimates of network parameters and mixing proportions. By using variational methods and minorization-maximization (MM) techniques, our method has appealing scalability for large-scale time-evolving networks. The power of our method is demonstrated in simulation studies and empirical applications to international trade networks and the collaboration networks of a large American research university.

- **11.Dynamic Community Detection Based on Network Structural Perturbation and Topological Similarity (Journal of Statistical Mechanics 2017)**
  - Peizhuo Wang, Lin Gao and Xiaoke Ma 
  - [[Paper]](https://iopscience.iop.org/article/10.1088/1742-5468/2017/1/013401/meta)
  - [[Matlab Reference]](https://github.com/WPZgithub/ESPRA)

Community detection in dynamic networks has been extensively studied since it sheds light on the structure-function relation of the overall complex systems. Recently, it has been demonstrated that the structural perturbation in static networks is excellent in characterizing the topology. In order to investigate the perturbation structural theory in dynamic networks, we extend the theory by considering the dynamic variation information between networks of consecutive time. Then a novel similarity is proposed by combing structural perturbation and topological features. Finally, we present an evolutionary clustering algorithm to detect dynamic communities under the temporal smoothness framework. Experimental results on both artificial and real dynamic networks demonstrate that the proposed similarity is promising in dynamic community detection since it improves the clustering accuracy compared with state-of-the-art methods, indicating the superiority of the presented similarity measure.

- **12.RDYN⁠: Graph Benchmark Handling Community Dynamics  (Arxiv 2017)**
  - Giulio Rossetti
  - [[Paper]](https://academic.oup.com/comnet/article/5/6/893/3925036)
  - [[Python Reference]](https://github.com/GiulioRossetti/RDyn)

Graph models provide an understanding of the dynamics of network formation and evolution; as a direct consequence, synthesizing graphs having controlled topology and planted partitions has been often identified as a strategy to describe benchmarks able to assess the performances of community discovery algorithm. However, one relevant aspect of real-world networks has been ignored by benchmarks proposed so far: community dynamics. As time goes by network communities rise, fall and may interact with each other generating merges and splits. Indeed, during the last decade dynamic community discovery has become a very active research field: in order to provide a coherent environment to test novel algorithms aimed at identifying mutable network partitions we introduce RDYN⁠, an approach able to generates dynamic networks along with time-dependent ground-truth partitions having tunable quality.

- **13.Sequential Detection of Temporal Communities by Estrangement Confinement (Scientific Reports 2012)**
  - Vikas Kawadia and Sameet Sreenivasan
  - [[Paper]](https://www.nature.com/articles/srep00794)
  - [[Python Reference]](https://github.com/kawadia/estrangement)
  
- **14.Detection of overlapping communities in dynamical social networks (IEEE SocialCom 2010)**
  - Rémy Cazabet, Frédéric Amblard, Chihab Hanachi
  - [[Paper]](http://cazabetremy.fr/Publications_files/iLCD%20SocialCom%20longVersion.pdf)
  - [[Java Reference]](http://cazabetremy.fr/rRessources/iLCD.html)
  
- **15.GraphScope: Parameter-Free Mining of Large Time-Evolving Graphs (KDD 2007)**
  - Jimeng Sun, Christos Faloutsos, Spiros Papadimitriou, and Philip S. Yu
  - [[Paper]](https://dl.acm.org/citation.cfm?id=1281266)
  - [[Java Reference]](https://github.com/sarovios/social-graph-cluster)

How can we find communities in dynamic networks of socialinteractions, such as who calls whom, who emails whom, or who sells to whom? How can we spot discontinuity time-points in such streams of graphs, in an on-line, any-time fashion? We propose GraphScope, that addresses both problems, using information theoretic principles. Contrary to the majority of earlier methods, it needs no user-defined parameters. Moreover, it is designed to operate on large graphs, in a streaming fashion. We demonstrate the efficiency and effectiveness of our GraphScope on real datasets from several diverse domains. In all cases it produces meaningful time-evolving patterns that agree with human intuition.
