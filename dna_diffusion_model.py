"""
DNA-Diffusion Model Wrapper
Singleton class to handle model loading and sequence generation
"""

import os
import sys
import torch
import numpy as np
import logging
from typing import Optional, Dict, List
import time

logger = logging.getLogger(__name__)

class DNADiffusionModel:
    """Singleton wrapper for DNA-Diffusion model"""
    _instance = None
    _initialized = False
    
    # Cell type mapping from simple names to dataset identifiers
    CELL_TYPE_MAPPING = {
        'K562': 'K562_ENCLB843GMH',
        'GM12878': 'GM12878_ENCLB441ZZZ',
        'HepG2': 'HepG2_ENCLB029COU',
        'hESCT0': 'hESCT0_ENCLB449ZZZ'
    }
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the model (only runs once due to singleton pattern)"""
        if not self._initialized:
            self._initialize()
            self._initialized = True
    
    def _initialize(self):
        """Load model and setup components"""
        try:
            logger.info("Initializing DNA-Diffusion model...")
            
            # Add DNA-Diffusion to path
            dna_diffusion_path = os.path.join(os.path.dirname(__file__), 'DNA-Diffusion')
            if os.path.exists(dna_diffusion_path):
                sys.path.insert(0, os.path.join(dna_diffusion_path, 'src'))
            
            # Import DNA-Diffusion components
            from dnadiffusion.models.pretrained_unet import PretrainedUNet
            from dnadiffusion.models.diffusion import Diffusion
            from dnadiffusion.data.dataloader import get_dataset_for_sampling
            
            # Load pretrained model from HuggingFace
            logger.info("Loading pretrained model from HuggingFace...")
            self.model = PretrainedUNet.from_pretrained("ssenan/DNA-Diffusion")
            
            # Move to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Initialize diffusion sampler with the model
            self.diffusion = Diffusion(
                model=self.model,
                timesteps=50,
                beta_start=0.0001,
                beta_end=0.2
            )
            
            # Ensure output_attention is set to False initially
            if hasattr(self.model, 'output_attention'):
                self.model.output_attention = False
            if hasattr(self.model.model, 'output_attention'):
                self.model.model.output_attention = False
            
            # Setup dataset for sampling
            data_path = os.path.join(dna_diffusion_path, "data/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt")
            saved_data_path = os.path.join(dna_diffusion_path, "data/encode_data.pkl")
            
            # Get dataset info
            train_data, val_data, cell_num_list, numeric_to_tag_dict = get_dataset_for_sampling(
                data_path=data_path,
                saved_data_path=saved_data_path,
                load_saved_data=True,
                debug=False,
                cell_types=None  # Load all cell types
            )
            
            # Store dataset info
            self.train_data = train_data
            self.val_data = val_data
            self.cell_num_list = cell_num_list
            self.numeric_to_tag_dict = numeric_to_tag_dict
            
            # Get available cell types
            self.available_cell_types = [numeric_to_tag_dict[num] for num in cell_num_list]
            logger.info(f"Available cell types: {self.available_cell_types}")
            
            # Warm up the model with a test generation
            logger.info("Warming up model...")
            self._warmup()
            
            logger.info("Model initialization complete!")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            self.model = None
            self.diffusion = None
            self.dataset = None
            raise
    
    def _warmup(self):
        """Warm up the model with a test generation"""
        try:
            # Generate one sequence for the first available cell type
            if self.available_cell_types:
                cell_type = list(self.CELL_TYPE_MAPPING.keys())[0]
                self.generate(cell_type, guidance_scale=1.0)
        except Exception as e:
            logger.warning(f"Warmup generation failed: {str(e)}")
    
    def is_ready(self) -> bool:
        """Check if model is loaded and ready"""
        return self.model is not None and self.diffusion is not None and self.train_data is not None
    
    def generate(self, cell_type: str, guidance_scale: float = 1.0) -> Dict[str, any]:
        """
        Generate a DNA sequence for the specified cell type
        
        Args:
            cell_type: Simple cell type name (K562, GM12878, HepG2, hESCT0)
            guidance_scale: Guidance scale for generation (1.0-10.0)
            
        Returns:
            Dict with 'sequence' (200bp string) and 'metadata'
        """
        if not self.is_ready():
            raise RuntimeError("Model is not initialized")
        
        # Validate inputs
        if cell_type not in self.CELL_TYPE_MAPPING:
            raise ValueError(f"Invalid cell type: {cell_type}. Must be one of {list(self.CELL_TYPE_MAPPING.keys())}")
        
        if not 1.0 <= guidance_scale <= 10.0:
            raise ValueError(f"Guidance scale must be between 1.0 and 10.0, got {guidance_scale}")
        
        # Map to full cell type identifier
        full_cell_type = self.CELL_TYPE_MAPPING[cell_type]
        
        # Find the numeric index for this cell type
        tag_to_numeric = {tag: num for num, tag in self.numeric_to_tag_dict.items()}
        
        # Find matching cell type (case-insensitive partial match)
        cell_type_numeric = None
        for tag, num in tag_to_numeric.items():
            if full_cell_type.lower() in tag.lower() or tag.lower() in full_cell_type.lower():
                cell_type_numeric = num
                logger.info(f"Matched '{full_cell_type}' to '{tag}'")
                break
        
        if cell_type_numeric is None:
            raise ValueError(f"Cell type {full_cell_type} not found in dataset. Available: {list(self.numeric_to_tag_dict.values())}")
        
        try:
            logger.info(f"Generating sequence for {cell_type} (guidance={guidance_scale})...")
            start_time = time.time()
            
            # For now, use simple generation without classifier-free guidance
            # TODO: Fix classifier-free guidance implementation
            sequence = self._generate_simple(cell_type_numeric, guidance_scale)
            
            generation_time = time.time() - start_time
            logger.info(f"Generated sequence in {generation_time:.2f}s")
            
            return {
                'sequence': sequence,
                'metadata': {
                    'cell_type': cell_type,
                    'full_cell_type': full_cell_type,
                    'guidance_scale': guidance_scale,
                    'generation_time': generation_time,
                    'sequence_length': len(sequence)
                }
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
    
    def _generate_simple(self, cell_type_idx: int, guidance_scale: float) -> str:
        """Simple generation using the diffusion model's sample method"""
        with torch.no_grad():
            # For guidance_scale = 1.0, use simple generation without classifier-free guidance
            if guidance_scale == 1.0:
                # Create initial noise
                img = torch.randn((1, 1, 4, 200), device=self.device)
                
                # Simple denoising loop without guidance
                for i in reversed(range(self.diffusion.timesteps)):
                    t = torch.full((1,), i, device=self.device, dtype=torch.long)
                    
                    # Get model prediction with classes
                    classes = torch.tensor([cell_type_idx], device=self.device, dtype=torch.long)
                    noise_pred = self.model(img, time=t, classes=classes)
                    
                    # Denoising step
                    betas_t = self.diffusion.betas[i]
                    sqrt_one_minus_alphas_cumprod_t = self.diffusion.sqrt_one_minus_alphas_cumprod[i]
                    sqrt_recip_alphas_t = self.diffusion.sqrt_recip_alphas[i]
                    
                    # Predict x0
                    model_mean = sqrt_recip_alphas_t * (img - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)
                    
                    if i == 0:
                        img = model_mean
                    else:
                        posterior_variance_t = self.diffusion.posterior_variance[i]
                        noise = torch.randn_like(img)
                        img = model_mean + torch.sqrt(posterior_variance_t) * noise
                
                final_image = img[0]  # Remove batch dimension
            else:
                # Use the diffusion model's built-in sample method with guidance
                # This requires proper context mask handling which is complex
                # For now, fall back to simple generation
                logger.warning(f"Guidance scale {guidance_scale} not fully implemented, using simple generation")
                return self._generate_simple(cell_type_idx, 1.0)
            
            # Convert to sequence
            final_array = final_image.cpu().numpy()
            sequence = self._array_to_sequence(final_array)
            
            return sequence
    
    def _array_to_sequence(self, array: np.ndarray) -> str:
        """Convert model output array to DNA sequence string"""
        # Get nucleotide mapping
        nucleotides = ['A', 'C', 'G', 'T']
        
        # array shape is (1, 4, 200) - channels, nucleotides, sequence_length
        # Reshape to (4, 200) and get argmax along nucleotide dimension
        array = array.squeeze(0)  # Remove channel dimension -> (4, 200)
        indices = np.argmax(array, axis=0)  # Get max nucleotide for each position
        
        # Convert indices to nucleotides
        sequence = ''.join(nucleotides[int(idx)] for idx in indices)
        
        return sequence
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model"""
        if not self.is_ready():
            return {'status': 'not_initialized'}
        
        return {
            'status': 'ready',
            'device': str(self.device),
            'cell_types': list(self.CELL_TYPE_MAPPING.keys()),
            'full_cell_types': self.available_cell_types,
            'model_name': 'ssenan/DNA-Diffusion',
            'sequence_length': 200,
            'guidance_scale_range': [1.0, 10.0]
        }


# Convenience functions for direct usage
_model_instance = None

def get_model() -> DNADiffusionModel:
    """Get or create the singleton model instance"""
    global _model_instance
    if _model_instance is None:
        _model_instance = DNADiffusionModel()
    return _model_instance

def generate_sequence(cell_type: str, guidance_scale: float = 1.0) -> str:
    """Generate a DNA sequence (convenience function)"""
    model = get_model()
    result = model.generate(cell_type, guidance_scale)
    return result['sequence']