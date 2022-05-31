# Diffusion-Model-Playgraound
**THE GOAL: manipulate sigma times z during sampling step**

Source git: [DiffusionClip](https://github.com/gwang-kim/DiffusionCLIP)

## Colab Playground
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1D_-8jELFZDQN-53St-Juov_hE8MHm3o0#scrollTo=q0WAU_sCp3xf)

## File Modification
* Delete *parse_args_and_config* and *main* functions of ``main.py``.

* Edit ``utils\diffusion_utils.py`` *denoising_step* function.
  
  * Create a new *add_var* parameter to manipulate **sigma times z**. If *add_var == False*, then no noise variance added during DDPM sampling step. *add_var == True* maintains the original DDPM sampling function.

* Change ``diffusionclip.py`` to ``ourddpm.py``; edit all *denoising_step* function correspondingly in the file.
