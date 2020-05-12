# LowProFool

LowProFool is an algorithm that generates imperceptible adversarial examples

This GitHub hosts the code for the following paper:

Ballet, V., Renard, X., Aigrain, J., Laugel, T., Frossard, P., & Detyniecki, M. (2019). Imperceptible Adversarial Attacks on Tabular Data. NeurIPS 2019 Workshop on Robust AI in Financial Services: Data, Fairness, Explainability, Trustworthiness, and Privacy (Robust AI in FS 2019), Vancouver, Canada
https://arxiv.org/abs/1911.03274

### Adverse.py

Contains the implementation of LowProFool [[1]](about:blank) along with an modifier version of DeepFool [[2]](about:blank) that handles tabular datasets.

### Metrics.py

Implements metrics introduced in [[1]](about:blank)

### Playground.ipynb

A demo python notebook to generate adversarial examples on the German Credit dataset and compare the results to DeepFool

## Reference
[1] Bibtex
[2] S. Moosavi-Dezfooli, A. Fawzi, P. Frossard: DeepFool: a simple and accurate method to fool deep neural networks. In Computer Vision and Pattern Recognition (CVPR ’16), IEEE, 2016.
