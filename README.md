# julia_class_project
## Implementing a Vision Transformer (ViT) in Julia!

Until recently, the best performing models for image classification had been convolutional neural networks (CNNs) introduced in [LeCun et al. (1998)](https://ieeexplore.ieee.org/abstract/document/726791). Nowadays, **transformer** architectures have been shown to have similar to better performance. One such model, called Vision Transformer by [Dosovitskiy et al. (2020)](https://arxiv.org/abs/2010.11929) splits up images into regularly sized patches. The patches are treated as a sequence and attention weights are learned as in a standard transformer model.

![ViT Model](https://github.com/qsimeon/julia_class_project/blob/main/figures/vit_architecture.jpg?raw=true)

---

The **Transformer** architecture, introduced in the paper _Attention Is All You Need_ by [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762), is the most ubiquitous neural network architecture in modern machine learning. Its parallelism and scalability to large problems has seen it adopted in domains beyong those it was traditionally considered for (sequential data). 

![Transformer Model](https://github.com/qsimeon/julia_class_project/blob/main/figures/transformer_architecture.jpg?raw=true)

**NOTE:** We adapt/borrow a lot of material/concepts from
[Torralba, A., Isola, P., & Freeman, W. T. (2021, December 1). _Foundations of Computer Vision_. MIT Press; The MIT Press, Massachusetts Institute of Technology.](https://mitpress.mit.edu/9780262048972/foundations-of-computer-vision/)

---

Potentially useful packages:
- https://juliaml.github.io/MLDatasets.jl/stable/
- https://enzyme.mit.edu/julia/stable/
- https://lux.csail.mit.edu/stable/

