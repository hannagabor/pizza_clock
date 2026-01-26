from pizza_clock.models import Model
import torch as t


def compute_gradient_similarity(model: Model) -> float:
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
