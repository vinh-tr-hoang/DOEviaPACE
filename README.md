# Bayesian Experimental Design via Projection-based Approximation of the Conditional Expectation (PACE) 
This repository hosts the codebase associated with the research paper 
"[Scalable method for Bayesian experimental design without integrating over posterior distribution](https://arxiv.org/abs/2306.17615)" by Vinh Hoang, Luis Espath,
Sebastian Krumscheid, and Raúl Tempone.
The article presents an efficient computational method solving A-optimal Bayesian design of experiments problems
where the observational model is based on partial differential equations and is computationally expensive to evaluate.
In this approach, we derive the expected conditional variance from the variance of the conditional expectation by
leveraging the law of total variance. Further, the orthogonal projection property is employed to approximate the conditional expectation.

Within this implementation, an Artificial Neural Network (ANN) is harnessed to approximate the complex non-linear conditional expectation.
To address the challenge of continuous experimental design parameters, we have integrated the training process of the ANN into the minimization of the expected conditional variance. 
In particular, we propose a non-local approximation of the conditional expectation and employ transfer learning to significantly reduce the number of evaluations needed for the observation model.

There are two demos implemented in this repository:

- Linear-Gaussian case, and
- Electrical impedance tomography (eit).

## Citation
```
@article{hoang_scalable_2025,
	title = {Scalable {Method} for {Bayesian} {Experimental} {Design} without {Integrating} over {Posterior} {Distribution}},
	volume = {13},
	issn = {2166-2525},
	url = {https://epubs.siam.org/doi/10.1137/23M1603364},
	doi = {10.1137/23M1603364},
	number = {1},
	journal = {SIAM/ASA J. Uncertainty Quantification},
	author = {Hoang, Vinh and Espath, Luis and Krumscheid, Sebastian and Tempone, Raúl},
	year = {2025},
	pages = {114--139},
}
```

## Authors

This repository was developed by [Vinh Hoang](mailto:hoang.tr.vinh@gmail.com).
For any further queries or issues related to the code, please contact the author.

## Dependencies
- numpy
- matplotlib
- tensorflow 
- sklearn
- joblib 
- seaborn
- pandas

## License
[License Information](https://github.com/vinh-tr-hoang/DOEviaPACE/blob/main/LICENSE)
