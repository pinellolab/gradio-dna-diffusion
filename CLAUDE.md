# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Gradio-based web application that visualizes DNA sequence generation as an interactive slot machine. The project demonstrates synthetic regulatory DNA sequence generation using the DNA-Diffusion model from Pinello Lab.

## Implementation Details

### Architecture
- **Backend**: Gradio app with DNA-Diffusion model integration
- **Frontend**: Custom HTML/JavaScript slot machine interface
- **Communication**: postMessage API between iframe and Gradio

### Key Components
1. **app.py**: Main Gradio application
2. **dna_diffusion_model.py**: Singleton wrapper around DNA-Diffusion model
3. **dna-slot-machine.html**: Interactive slot machine visualization
4. **requirements.txt**: Python dependencies

### Model Integration
- Model loads from HuggingFace: `ssenan/DNA-Diffusion`
- Supports cell types: K562, GM12878, HepG2, hESCT0
- Generates 200bp regulatory sequences
- Uses simplified generation (guidance_scale=1.0) for stability

### UI Features
- Real-time spinning animation during generation
- 200 slot reels representing each base pair
- Pull lever or click button to generate
- Color-coded nucleotides (A=green, T=red, C=blue, G=yellow)

## Development Notes

### Running Locally
```bash
# With DNA-Diffusion installed
DNA-Diffusion/.venv/bin/python app.py

# Mock mode (no model)
python app.py
```

### Important Reminders
- ALWAYS prefer editing existing files over creating new ones
- NEVER create documentation files unless explicitly requested
- The model requires DNA-Diffusion to be installed via uv
- First generation includes model warm-up time