# **Hamiltonian Dynamics for Real-World Shape Interpolation**

* Implementation of "Hamiltonian Dynamics for Real-World Shape Interpolation" (ECCV 20). Authors: Eisenberger, Cremers. arXiv: https://arxiv.org/abs/2004.05199.
* Vanilla implementation of "Divergence-Free Shape Correspondence by Deformation" (SGP 19). Authors: Eisenberger, Lähner, Cremers. https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.13785

<details><summary>Requirements</summary>

* Python Version >= 3.6
* PyTorch 1.4
* CUDA 9.2 or 10.1
* torch-geometric library

</details>


# Run the test script:
* python3 main_interpolation.py
* You can specify to compute an interpolation with one of the other methods in the main script in main_interpolation.py

# Use your own examples:
* For different input shapes, see tools/shape_utils.py to load shapes with an equivalent meshing and tools/partial_shapes.py for shapes with incompatible meshing or partial shapes.
* You can also prepare the input pair such that they have compatible vertices, e.g. by transferring the meshing with the correspondences beforehand.
* Compute interpolations for a complete dataset with the routine "main_dataset" in main_interpolation.py
* Evaluate the error metrics from the paper with the scripts in interpolation/eval_interpolation.py
