# LLC analysis of grokking on modular arithmetic

Disclaimer: This project is WIP, I am currently just in the process of reimplementing the Pizza and the Clock paper.

## Goals of the project

* Practice coding something in PyTorch from scratch
* Make some progress on a developmental interpretability [project](https://timaeus.co/projects/grokking) from Timaeus.

## Plan

1. Reimplement some experiments from [The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks](https://arxiv.org/abs/2306.17844). (They shared the code, but as mentioned earlier, I would like to write something from scratch.)
2. Make sure we can searate pizza and clock solutions.
3. Track LLC during training. Can we differentiate Pizza and Clock solutions based on the LLC?
4. Investigate other scenarios. (Expand plan.)


## Journal
* Reimplemented the models from the paper, they kind of work. There is a sudden drop in validation loss after 15000 steps, but the models do not reach 100% validation accuracy. Why?
![alt text](screenshots/image.png)
![alt text](screenshots/image-1.png)

* What differences are there between my implementation and theirs?
    *  I used `nn.Embedding`, which initializes the weight matrix from $\mathcal{N}(0,1)$, while tha paper used `torch.randn(d_model, d_vocab)/np.sqrt(d_model)` for weight init. The difference is the scaling by the square root of `d_model`.
    * For unembedding, I used `nn.Linear`. The first problem is that I also added a bias. The second in the initialization again. `nn.Linear` uses a uniform distribution, the original paper uses a scaled uniform distribution. `torch.randn(d_model, d_vocab)/np.sqrt(d_vocab))`
    * In general, they always initialize things such that the output of the layer will start with 1 std.
    * It is important that the weight decay is $2$, not $1e-2$...
    * I didn't have a scheduler that ramps up the learning rate to the given value gradually, during the first 10 steps. Not sure if this is important.
    * I had other bugs, like sometimes forgetting to use `nn.Parameter`.
* Now I have validation accuracy $1$! There is a small delay in the validation loss dropping to $0$ after the training loss has dropped to $0$.This is grokking, but not as pronounced as in the Nanda paper.
![alt text](screenshots/train_loss_after_fixes.png)
![alt text](screenshots/val_loss_after_fixes.png)
* Hm, Nanda et al took $30$% of the data, not $80$, so grokking was more difficult than here with $80$. I didn't run these ecperiments for long, but I could see the grokking for some of the trajectories and the validation accuracy not dropping for others. 
![alt text](screenshots/p113,training_ratio0.3.png)
* From an LLC research point of view, it would be best to find parameters where grokking is clearly separated from memorization, but it doesn't take too long to grok. Although, using the setup from the Pizza and the Clock paper, we are at least sure we have pizza and clock exmaples.
