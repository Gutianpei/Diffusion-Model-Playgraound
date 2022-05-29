# Diffusion-Model-Playgraound
**THE GOAL: manipulate sigma times z during sampling step**
Source git: [DiffusionClip](https://github.com/gwang-kim/DiffusionCLIP)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1D_-8jELFZDQN-53St-Juov_hE8MHm3o0#scrollTo=q0WAU_sCp3xf)

## File Modification
* Delete parse_args_and_config and main functions of ``main.py``.

* Edit ``utils.diffusion_utils denoising_step`` function.

* Change ``diffusionclip.py`` to ``ourddpm.py``.
