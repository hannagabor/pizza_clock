# LLC analysis of grokking on modular arithmetic

Disclaimer: This project is WIP, I am currently just in the process of reimplementing the Pizza and the Clock paper.

## Goals of the project

* Practice coding something in PyTorch from scratch
* Make some progress on a developmental interpretability [project](https://timaeus.co/projects/grokking) from Timaeus.

## Plan

1. Reimplement some experiments from [The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks](https://arxiv.org/abs/2306.17844).
2. Track LLC during training. Can we differentiate Pizza and Clock solutions based on the LLC?
3. Investigate other scenarios.


## Journal
* Reimplemented the models from the paper, they kind of work. There is a sudden drop in validation loss after 15000 steps, but the models do not reach 100% validation accuracy. Why?
![alt text](screenshots/image.png)
![alt text](screenshots/image-1.png)


