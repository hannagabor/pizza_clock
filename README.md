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

* What differences are there between my implementation and theirs?
    *  I used `nn.Embedding`, which initializes the weight matrix from $\mathcal{N}(0,1)$, while tha paper used `torch.randn(d_model, d_vocab)/np.sqrt(d_model)` for weight init. The difference is the scaling by the square root of `d_model`.
    * For unembedding, I used `nn.Linear`. The first problem is that I also added a bias. The second in the initialization again. `nn.Linear` uses a uniform distribution, the original paper uses a scaled uniform distribution. `torch.randn(d_model, d_vocab)/np.sqrt(d_vocab))`
    * In general, they always initialize things such that the output of the layer will start with 1 std.
    * I think it is important that the weight decay is $2$, not $1e-2$...
    * I didn't have a scheduler that ramps up the learning rate to the given value gradually, during the first 10 steps. Not sure if this is important.
    * I had other bugs, like sometimes forgetting to use `nn.Parameter`.
