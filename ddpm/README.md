
- `ddpm.py`: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).

Notes:
- `sample()` computes a single sampling step of Algorithm 2.
    - Computing the whole algorithm requires some way of handling the whole model outputting the required `eps_theta_t` (i.e. passing it as argument, as in https://github.com/crowsonkb/k-diffusion).
    - I chose to handle a single step, to avoid some coupling between the sampler and the NN giving $\epsilon_\theta(\mathbf{x}_t, t)$ (passing the model $\implies$ passing all its required inputs. If the model's inputs change, then we'd need to change 1) the signature of the model's `forward()` method + its code, 2) the signature of `sample()` + its code (since we'd be calling `model()` from inside). Not neat.
    - This is in the same spirit as the design choice taken in 🤗diffusers, where `step()` computes a single sampling step in the loop (see details here: https://github.com/huggingface/diffusers/issues/1308#issuecomment-1318361179)
- In 🤗diffusers, the number of timesteps during sampling (reverse process) is allowed to be less than the number of timesteps when adding noise (forward process).
    - I guess this comes from the insights provided by the [DDIM paper](https://arxiv.org/pdf/2010.02502.pdf).
    - In DDPM I didn't see anything about accelerating the generation process, so I left it as it is: `num_inferences_steps =  num_train_timesteps = max_num_timesteps`.
- To do:
    - Add pipeline code computing the whole algorithm 2.
    - Add training code.