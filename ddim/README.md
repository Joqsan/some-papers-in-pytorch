
- `ddim.py`: [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502).

Notes:
- `sample()` computes a single sampling step.
    - Computing the whole algorithm requires some way of handling the whole model outputting the required `eps_theta_t` (i.e. passing it as argument, as in https://github.com/crowsonkb/k-diffusion).
    - I chose to handle a single step, to avoid some coupling between the sampler and the NN giving $\epsilon_\theta(\mathbf{x}_t, t)$ (passing the model $\implies$ passing all its required inputs. If the model's inputs change, then we'd need to change 1) the signature of the model's `forward()` method + its code, 2) the signature of `sample()` + its code (since we'd be calling `model()` from inside). Not neat.
    - This is in the same spirit as the design choice taken in 🤗diffusers, where `step()` computes a single sampling step in the loop (see details here: https://github.com/huggingface/diffusers/issues/1308#issuecomment-1318361179).
- DDIM when $\eta = 0$.
    - The forward process becomes deterministic.
- Here we allowed for accelerated generation process: `num_inferences_steps <=  num_train_timesteps = max_num_timesteps`.
- To do:
    - Add pipeline code computing the whole sampling algorithm.
    - Add training code.