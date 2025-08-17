# Residual Reservoir Memory Networks
[[paper]](https://arxiv.org/pdf/2508.09925)

Official code for _Residual Reservoir Memory Networks_, IJCNN (2025) paper.

### Abstract
We introduce a novel class of untrained Recurrent Neural Networks (RNNs) within the Reservoir Computing (RC) paradigm, called Residual Reservoir Memory Networks (ResRMNs). ResRMN combines a linear memory reservoir with a non-linear reservoir, where the latter is based on residual orthogonal connections along the temporal dimension for enhanced long-term propagation of the input. The resulting reservoir state dynamics are studied through the lens of linear stability analysis, and we investigate diverse configurations for the temporal residual connections. The proposed approach is empirically assessed on time-series and pixel-level 1-D classification tasks. Our experimental results highlight the advantages of the proposed approach over other conventional RC models.

### Setup
```
conda create -n residualrmn python=3.12
conda activate residualrmn
pip install -r requirements.txt
pip install -e .
```

### Run
See the [experiments guide](./experiments/README.md) for instructions on reproducing the experiments in the paper, or running your own.

### Citation
If you use the code in this repository, consider citing our paper:
```
@article{pinna2025residual,
  title={Residual Reservoir Memory Networks},
  author={Pinna, Matteo and Ceni, Andrea and Gallicchio, Claudio},
  journal={arXiv preprint arXiv:2508.09925},
  year={2025}
}
```
