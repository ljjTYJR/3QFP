# 3QFP
3QFP: Efficient neural implicit surface reconstruction using Tri-Quadtrees and Fourier feature Positional encoding [ICRA24]

# Overview

<p align="center">
  <a href="">
    <img src="./assets/teaser.jpg" alt="teaser" width="100%">
  </a>
</p>

**Overview of our method.**

We represent the scene with three planar quadtrees $\mathcal{M}_{i}^{\ell}$, $i \in \{XZ,YZ,XY\}$ and $\ell$ represents the quadtree depth. We store features in the deepest $H$ levels of resolution of quadtrees. When querying for a point $\mathbf{p}$, we project it onto planar quadtrees to identify the node containing $\mathbf{p}$ at the level $\ell$. The feature of $\mathbf{p}$ is then calculated by bilinear interpolation based on the queried location and vertex features. We add features at the same level and concatenate among different levels. Concatenated with the positional encoding $\gamma(\mathbf{p})$, $\mathbf{p}$'s feature~($\Phi(\mathbf{p})$) is fed into a small MLP~($\mathcal{F}_\Theta$) to predict the SDF value. The learnable features stored in the quadtree nodes and the network parameters are optimized in real-time using the loss function $\mathcal{L}_{\text{bce}}$. The learnable feature vectors have length $d$ and the positional encoding feature vector has length $6m$.

# Installation
The code will be released next month