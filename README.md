<div align="center">

# Residual Reservoir Memory Networks

[![arXiv](https://img.shields.io/badge/arXiv-2508.09925-b31b1b.svg)](https://arxiv.org/abs/2508.09925)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow)](https://huggingface.co/papers/2508.09925)

[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/nennomp/research-code-template)
![code-quality](https://github.com/nennomp/residualrmn/actions/workflows/code-quality.yml/badge.svg)

</div>

This repository contains the official code for the paper:

```
Residual Reservoir Memory Networks,
Matteo Pinna, Andrea Ceni, Claudio Gallicchio
International Joint Conference on Neural Networks (IJCNN), 2025.
```

[![arXiv](https://img.shields.io/badge/arXiv-2508.09925-b31b1b.svg)](https://arxiv.org/abs/2508.09925)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow)](https://huggingface.co/papers/2508.09925)

## Abstract
We introduce a novel class of untrained Recurrent Neural Networks (RNNs) within the Reservoir Computing (RC) paradigm, called Residual Reservoir Memory Networks (ResRMNs). ResRMN combines a linear memory reservoir with a non-linear reservoir, where the latter is based on residual orthogonal connections along the temporal dimension for enhanced long-term propagation of the input. The resulting reservoir state dynamics are studied through the lens of linear stability analysis, and we investigate diverse configurations for the temporal residual connections. The proposed approach is empirically assessed on time-series and pixel-level 1-D classification tasks. Our experimental results highlight the advantages of the proposed approach over other conventional RC models.

<div align="center">
<img src="assets/figure-1.png?raw=true" alt="Model" title="Model">
</div>
<figcaption><em><strong>
Architecture of ResRMN.</strong> The model consists of two untrained components: (i) a linear memory reservoir driven by the external input $x$ and (ii) a non-linear residual reservoir driven by both the external input x and the output $m$ of the memory reservoir. The final output is fed to a simple (linear) readout layer trained in closed-form, which is the only trainable component.
</em></figcaption>

## Setup
To install the required dependencies:
```
conda create -n residualrmn python=3.12
conda activate residualrmn
pip install -e .
```

## Experiments
See the [experiments guide](./experiments/README.md) for instructions on reproducing the experiments in the paper, or running your own.

## Citation
If you use the model or code in this repository, consider citing our paper:
```
@article{pinna2025residual,
  title={Residual Reservoir Memory Networks},
  author={Pinna, Matteo and Ceni, Andrea and Gallicchio, Claudio},
  journal={arXiv preprint arXiv:2508.09925},
  year={2025}
}
```
