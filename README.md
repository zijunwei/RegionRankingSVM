## RRSVM_pytorch: An End-to-End PyTorch Implementation of Region Ranking SVM

This is the combined repository implementing the following two papers:

1. **Region Ranking SVM for Image Classification** (CVPR 2016) — Z. Wei, M. Hoai.
   [paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Wei_Region_Ranking_SVM_CVPR_2016_paper.pdf)
2. **Improving Human Action Recognition by Non-action Classification** (NeurIPS 2016).
   [paper](https://proceedings.neurips.cc/paper_files/paper/2016/file/a0e2a2c563d57df27213ede1ac4ac780-Paper.pdf)

## Related Repositories

- [`zijunwei/SDR`](https://github.com/zijunwei/SDR) *(private)* — original MATLAB
  implementation of RRSVM (CVPR 2016) and SDR (NeurIPS 2016) under `matlab/`,
  plus an early Python port under `python/`. Depends on IBM CPLEX. The
  `matlab/Region-Ranking-SVM-master/` subtree preserves the contents of the
  old standalone `zijunwei/Region-Ranking-SVM` repo (no longer on GitHub).
- [`zijunwei/ZFunc`](https://github.com/zijunwei/ZFunc) *(private)* — personal
  MATLAB helper functions used as a dependency by the MATLAB RRSVM code.

## TODO

- [ ] **Migrate environment management to [uv](https://docs.astral.sh/uv/)**
  so the project is easy to set up and run on a fresh machine.
  - [ ] Add `pyproject.toml` with pinned dependencies (PyTorch, torchvision, numpy, etc.).
  - [ ] Generate and commit a `uv.lock` for reproducible installs.
  - [ ] Replace the legacy Python 2.7 instructions below with a `uv sync` / `uv run` workflow.
- [ ] **Modernize the codebase to Python 3 and current PyTorch**
  - [ ] Port `RRSVM` and `SoftMaxRRSVM` C/CUDA extensions from the old
    `torch.utils.ffi` build (`build_RRSVM.py`, `build_SoftMaxRRSVM.py`) to
    the current `torch.utils.cpp_extension` API.
  - [ ] Update `Mnist_main.py` and other training scripts to Python 3 syntax.
- [ ] **Merge the two papers' code paths into a single, unified project**
  - [ ] Consolidate the CVPR 2016 (Region Ranking SVM) training/eval code.
  - [ ] Consolidate the NeurIPS 2016 (non-action classification / HICO / MPII) code
    under a shared module layout.
  - [ ] Shared data loaders, checkpoint format, and config system across both papers.
- [ ] **Reproducibility**
  - [ ] Document dataset download/preparation steps for MNIST, CIFAR-10, HICO, MPII.
  - [ ] Provide example `uv run` commands to reproduce the main results of each paper.
- [ ] **CI / sanity checks**
  - [ ] Add a smoke test that builds the extensions and runs a minimal forward/backward pass.

###  Description
The modules of **RRSVM** and **SoftMaxRRSVM** are implemented in the **RRSVM** directory.

The source code are saved in RRSVM_src and SoftMaxRRSVM_src respectively.

Some tests on the modules are saved in RRSVM/Tests

For each of the module, there are two implementations: the CPU version and the GPU version (XXX_cuda.c), the XXX_cuda.c calls the kernel file saved in cuda dir.



### How to start (legacy — pending uv migration, see TODO above):
1.  Download pytorch with python 2.7 and GPU support
2.  Compile the RRSVM module and SoftMaxRRSVM module by:
	```
	cd RRSVM
	python build_RRSVM.py
	python build_SoftMaxRRSVM.py
	```
3. Run Mnist_main.py to check if everything is correct.


### PS

the dataset is saved in ~\datasets\RRSVM_datasets,  if not available, it will download and create them.


### Progress:

#### Jan 12, 2018

**Sanity Check**
I checked the gradients & backpropgation of all the variantations of RRSVM in CPU and Cuda versions. 
All seem to be correct. The only error comes from the GradInput when the value is large. This is mostly
likely to be caused by the instabilities of float format. I checked the officially implemented conv2d
when using FloatTensor as input, it happens to have errors. But when I change the input variables to be
DoubleTensor, the errors are gone. 


**A possible Problem** In constrast to conv operations, in RRSVM, suppose the weight is a 1 by 4 vector. Then the first
is always corresponding to the largest value (since the input is sorted), it will receive the largest gradient change all the
time. Will this bring instabilities? Should we apply lower learning rates to the higher indices in w? 

At least the first thing to do is to reduce the learning rate of the whole RRSVM layer to see if it gets stable


**A possbile extension** We have been attempting to investigate the accuracy/speed/memory gain of RRSVM but received all 
negative gains. The possible next direction would be investigating the explaination/intepretibility gain... 
(See the paper: https://arxiv.org/pdf/1711.05611.pdf)
