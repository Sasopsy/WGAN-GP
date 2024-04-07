
# DCGAN with Wasserstein Loss and Gradient Penalty

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) enhanced with Wasserstein Loss and Gradient Penalty for stable training and improved generation quality.

## File Descriptions

- **`train.py`**: Main training script, orchestrating the model training with dataset preparation and training loops.
- **`dataset.py`**: Handles dataset loading and preprocessing, tailored for the DCGAN model requirements.
- **`model.py`**: Defines the DCGAN architecture, including both Generator and Discriminator models.
- **`utils.py`**: Provides utility functions and classes to support model training and data manipulation.

## References

- DCGAN Paper: ["Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"](https://arxiv.org/pdf/1511.06434.pdf)
- Wasserstein Loss Paper: ["Wasserstein GAN"](https://arxiv.org/pdf/1701.07875.pdf)
- Gradient Penalty Paper: ["Improved Training of Wasserstein GANs"](https://arxiv.org/pdf/1704.00028.pdf)
