# IST Project Release

The implementation of the IST algorithm for the 2-/3- layer google speech recognition.

Two implementations are provided:

- Centralized Parameter Server;


- Distributed Parameter Server.

If you use AWS deep learning AMI, you can follow the instructions below:

`` source activate pytorch_p36 ``

`` export GLOO_SOCKET_IFNAME=ens5 ``

or

``export GLOO_SOCKET_IFNAME=ens3``

Run the training, e.g. :

`` python distributed_2layer_subnet_centralized_ps_gloo.py --world-size 2 --rank 0``

``python distributed_2layer_subnet_centralized_ps_gloo.py --world-size 2 --rank 1``

If you find this code helpful, please cite this article:

````
@article{DBLP:journals/corr/abs-1910-02120,
  author    = {Binhang Yuan and
               Anastasios Kyrillidis and
               Christopher M. Jermaine},
  title     = {Distributed Learning of Deep Neural Networks using Independent Subnet
               Training},
  journal   = {CoRR},
  volume    = {abs/1910.02120},
  year      = {2019},
  url       = {http://arxiv.org/abs/1910.02120},
  archivePrefix = {arXiv},
  eprint    = {1910.02120},
  timestamp = {Wed, 09 Oct 2019 14:07:58 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1910-02120.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
````