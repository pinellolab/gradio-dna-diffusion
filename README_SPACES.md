---
title: DNA Diffusion Slot Machine
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app_spaces.py
pinned: false
license: apache-2.0
---

# DNA-Diffusion Slot Machine 🎰🧬

An interactive web application for generating cell type-specific DNA regulatory sequences using the DNA-Diffusion model from [Pinello Lab](https://pinellolab.org).

## Features

- 🎰 **Interactive Slot Machine Interface**: Watch 200 slots spin as DNA sequences are generated
- 🧬 **Cell Type-Specific**: Generate sequences for K562, GM12878, and HepG2 cell lines
- ⚡ **Real-time Animation**: Visual feedback during sequence generation
- 🎨 **Beautiful UI**: Retro-futuristic design with smooth animations

## Note

This is a demo version running in mock mode. For real DNA sequence generation:
1. Deploy with GPU enabled
2. Install DNA-Diffusion model dependencies
3. Use the full `app.py` instead of `app_spaces.py`

## Usage

1. Select a cell type (K562, GM12878, or HepG2)
2. Click GENERATE or pull the lever
3. Watch the slots spin!
4. View your generated 200bp regulatory sequence

## Citation

If you use this application in your research, please cite:

```bibtex
@article{dnadiffusion2024,
  title={DNA-Diffusion: Leveraging Generative Models for Controlling Chromatin Accessibility and Gene Expression via Synthetic Regulatory Elements},
  author={DaSilva, Lucas Ferreira and Senan, Simon and Patel, Zain Munir and others},
  journal={bioRxiv},
  year={2024},
  doi={10.1101/2024.02.01.578352}
}
```

## Links

- [GitHub Repository](https://github.com/pinellolab/gradio-dna-diffusion)
- [DNA-Diffusion Paper](https://www.biorxiv.org/content/10.1101/2024.02.01.578352v1)
- [Pinello Lab](https://pinellolab.org)