from pizza_clock.models import Model
import torch as t
from torch import Tensor
from jaxtyping import Float


def compute_gradient_symmetry(model: Model) -> float:
    p = model.config.p
    inputs = t.randint(0, p, size=(100, 2))
    out_logits = t.randint(0, p, size=(100,))

    gradient_cosines_total = 0
    for i, input_tensor in enumerate(inputs):
        model.zero_grad()
        model.token_embedding.zero_grad()

        logits = model.forward(input_tensor.unsqueeze(0))
        last_logits = logits[0, -1, :]
        target_logit = last_logits[out_logits[i]]
        target_logit.backward()

        first_token_grad, second_token_grad = model.token_embedding.output_gradients[0]
        gradient_cosine = t.nn.functional.cosine_similarity(
            first_token_grad.unsqueeze(0), second_token_grad.unsqueeze(0)
        ).item()
        gradient_cosines_total += gradient_cosine
    return gradient_cosines_total / len(inputs)


def compute_distance_irrelevance(model: Model) -> float:
    logit_matrix = get_logit_matrix(model)
    avg_std_per_row = logit_matrix.std(dim=1).mean().item()
    overall_std = logit_matrix.std().item()
    return avg_std_per_row / overall_std


def get_logit_matrix(model: Model) -> Float[Tensor, "p p"]:
    p = model.config.p
    input_tensor = t.tensor([[a, b] for a in range(p) for b in range(p)], dtype=t.long)
    correct_logit_index = t.tensor([(a + b) % p for a in range(p) for b in range(p)])
    logits = model(input_tensor)[:, -1, :]  # [batch size, p]
    correct_logits = logits[t.arange(p * p), correct_logit_index]
    logit_matrix = correct_logits.reshape(p, p)
    return logit_matrix
