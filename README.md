# PINEURODEs

PINEURODEs stands for Physics-Informed Neural ODEs. 

## Welcome

This project begins as a working environment for an MSc research project at the [Department of Chemical Engineering](https://www.imperial.ac.uk/chemical-engineering), in the [Complex Multiscale Systems group](https://www.imperial.ac.uk/complex-multiscale-systems/). 

Title of the project: *Data-driven modelling and prediction of complex systems*

### Project plan
The goal of this project is to use state-of-the-art artificial intelligence (AI) and data-driven frameworks to efficiently simulate and accurately predict complex systems. Our interest focuses on families of continuous deep architectures, which have recently re-emerged as **neural ordinary differential equations** (ODEs) and **physics-informed neural networks** (PINNs). 

The project tries to tackle the following problems (ordered by level of complexity):

* Model prototypes, e.g. the Lorenz system, with the help of neural ODEs. The Lorenz system exhibits non-trivial behaviour including transition to chaos, which makes the problem challenging, and hence interesting by construction.
* Reproduce the Kermack-McKendrick model of epidemiology, in particular the simple SIR (susceptible-infective-removed)
variant via neural ODEs. At this stage, after having used applied them for Lorenz's system, it should be possible to extend the framework to this model.
* Damaged image re-construction using Cahn-Hilliard (CH) image inpainting. Image inpainting consists of filling
damaged or missing areas of an image, with the ultimate objective of restoring it and making it appear as the true and
original image. Here we shall make use of CH as a prototypical system. CH allows for the formation of two phases separated by a smooth (“fuzzy” to adopt a term from AI) interface and hence it naturally allows for binary images.
* Social dynamics prediction, with special attention to epidemiology and/or fake-news spreading, by using
convolutional neural networks (e.g. FlowNet) in combination with simulations of either spatially-extended compartmental (SIR) models, or agent-based simulations results.

Basic questions to be addressed include: 

* How to select training data sets
* How to build the neural networks (and what framework will fit best our purposes)
* Extend the results from one-dimension (1D) to 2D, or even 3D.

## Contributing

Our style of code adheres to Google's standards ([google-styleguide](https://google.github.io/styleguide/pyguide.html)). We want to keep the source consistent, readable and easy to merge. For this reason we use rigid coding style and we expect all contributors to conform this guidelines. Please, use [.clang-format](.clang-format) to check your formatting.

## License 

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
