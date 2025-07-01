"""
DNA-Diffusion Gradio Application
Interactive DNA sequence generation with slot machine visualization
"""

import gradio as gr
import logging
import json
import os
from typing import Dict, Any, Tuple
import html

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import spaces for GPU decoration
try:
    import spaces
    SPACES_AVAILABLE = True
except ImportError:
    SPACES_AVAILABLE = False
    # Create a dummy decorator if spaces is not available
    class spaces:
        @staticmethod
        def GPU(duration=60):
            def decorator(func):
                return func
            return decorator

# Try to import model, but allow app to run without it for UI development
try:
    from dna_diffusion_model import DNADiffusionModel, get_model
    MODEL_AVAILABLE = True
    logger.info("DNA-Diffusion model module loaded successfully")
except ImportError as e:
    logger.warning(f"DNA-Diffusion model not available: {e}")
    MODEL_AVAILABLE = False

# Load the HTML interface
HTML_FILE = "dna-slot-machine.html"
if not os.path.exists(HTML_FILE):
    raise FileNotFoundError(f"HTML interface file '{HTML_FILE}' not found. Please ensure it exists in the same directory as app.py")

with open(HTML_FILE, "r") as f:
    SLOT_MACHINE_HTML = f.read()

class DNADiffusionApp:
    """Main application class for DNA-Diffusion Gradio interface"""
    
    def __init__(self):
        self.model = None
        self.model_loading = False
        self.model_error = None
        
    def initialize_model(self):
        """Initialize the DNA-Diffusion model"""
        if not MODEL_AVAILABLE:
            self.model_error = "DNA-Diffusion model module not available. Please install dependencies."
            return
        
        if self.model_loading:
            return
        
        self.model_loading = True
        try:
            logger.info("Starting model initialization...")
            self.model = get_model()
            logger.info("Model initialized successfully!")
            self.model_error = None
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            self.model_error = str(e)
            self.model = None
        finally:
            self.model_loading = False
    
    @spaces.GPU(duration=60)
    def generate_sequence(self, cell_type: str, guidance_scale: float = 1.0) -> Tuple[str, Dict[str, Any]]:
        """Generate a DNA sequence using the model or mock data"""
        
        # Use mock generation if model is not available
        if not MODEL_AVAILABLE or self.model is None:
            logger.warning("Using mock sequence generation")
            import random
            sequence = ''.join(random.choice(['A', 'T', 'C', 'G']) for _ in range(200))
            metadata = {
                'cell_type': cell_type,
                'guidance_scale': guidance_scale,
                'generation_time': 2.0,
                'mock': True
            }
            # Simulate generation time
            import time
            time.sleep(2.0)
            return sequence, metadata
        
        # Use real model
        try:
            result = self.model.generate(cell_type, guidance_scale)
            return result['sequence'], result['metadata']
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def handle_generation_request(self, cell_type: str, guidance_scale: float):
        """Handle sequence generation request from Gradio"""
        try:
            logger.info(f"Generating sequence for cell type: {cell_type}")
            sequence, metadata = self.generate_sequence(cell_type, guidance_scale)
            return sequence, json.dumps(metadata)
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Generation request failed: {error_msg}")
            return "", json.dumps({"error": error_msg})

# Create single app instance
app = DNADiffusionApp()

def create_demo():
    """Create the Gradio demo interface"""
    
    # CSS to hide backend controls and prevent scrolling
    css = """
    #hidden-controls { display: none !important; }
    .gradio-container { 
        overflow: hidden; 
        background-color: #000000 !important;
    }
    #dna-frame { overflow: hidden; position: relative; }
    body {
        background-color: #000000 !important;
    }
    """
    
    # JavaScript for handling communication between iframe and Gradio
    js = """
    function() {
        console.log('Initializing DNA-Diffusion Gradio interface...');
        
        // Set up message listener to receive requests from iframe
        window.addEventListener('message', function(event) {
            console.log('Parent received message:', event.data);
            
            if (event.data.type === 'generate_request') {
                console.log('Triggering generation for cell type:', event.data.cellType);
                
                // Update the hidden cell type input
                const radioInputs = document.querySelectorAll('#cell-type-input input[type="radio"]');
                radioInputs.forEach(input => {
                    if (input.value === event.data.cellType) {
                        input.checked = true;
                        // Trigger change event
                        input.dispatchEvent(new Event('change'));
                    }
                });
                
                // Small delay to ensure radio button update is processed
                setTimeout(() => {
                    document.querySelector('#generate-btn').click();
                }, 100);
            }
        });
        
        // Function to send sequence to iframe
        window.sendSequenceToIframe = function(sequence, metadata) {
            console.log('Sending sequence to iframe:', sequence);
            const iframe = document.querySelector('#dna-frame iframe');
            if (iframe && iframe.contentWindow) {
                try {
                    const meta = JSON.parse(metadata);
                    if (meta.error) {
                        iframe.contentWindow.postMessage({
                            type: 'generation_error',
                            error: meta.error
                        }, '*');
                    } else {
                        iframe.contentWindow.postMessage({
                            type: 'sequence_generated',
                            sequence: sequence,
                            metadata: meta
                        }, '*');
                    }
                } catch (e) {
                    console.error('Failed to parse metadata:', e);
                    // If parsing fails, still send the sequence
                    iframe.contentWindow.postMessage({
                        type: 'sequence_generated',
                        sequence: sequence,
                        metadata: {}
                    }, '*');
                }
            } else {
                console.error('Could not find iframe');
            }
        };
    }
    """
    
    with gr.Blocks(css=css, js=js, theme=gr.themes.Base()) as demo:
        
        # Hidden controls for backend processing
        with gr.Column(elem_id="hidden-controls", visible=False):
            cell_type_input = gr.Radio(
                ["K562", "GM12878", "HepG2"],
                value="K562",
                label="Cell Type",
                elem_id="cell-type-input"
            )
            guidance_input = gr.Slider(
                minimum=1.0,
                maximum=10.0,
                value=1.0,
                step=0.5,
                label="Guidance Scale",
                elem_id="guidance-input"
            )
            generate_btn = gr.Button("Generate", elem_id="generate-btn")
            
            sequence_output = gr.Textbox(label="Sequence", elem_id="sequence-output")
            metadata_output = gr.Textbox(label="Metadata", elem_id="metadata-output")
        
        # Main interface - the slot machine in an iframe
        # Escape the HTML content for srcdoc
        escaped_html = html.escape(SLOT_MACHINE_HTML, quote=True)
        iframe_html = f'<iframe srcdoc="{escaped_html}" style="width: 100%; height: 800px; border: none; display: block;"></iframe>'
        
        html_display = gr.HTML(
            iframe_html,
            elem_id="dna-frame"
        )
        
        # Wire up the generation
        generate_btn.click(
            fn=app.handle_generation_request,
            inputs=[cell_type_input, guidance_input],
            outputs=[sequence_output, metadata_output]
        ).then(
            fn=None,
            inputs=[sequence_output, metadata_output],
            outputs=None,
            js="(seq, meta) => sendSequenceToIframe(seq, meta)"
        )
        
        # Initialize model on load
        demo.load(
            fn=app.initialize_model,
            inputs=None,
            outputs=None
        )
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_demo()
    
    # Parse any command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="DNA-Diffusion Gradio App")
    parser.add_argument("--share", action="store_true", help="Create a public shareable link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the app on")
    args = parser.parse_args()
    
    # For Hugging Face Spaces deployment
    import os
    if os.getenv("SPACE_ID"):
        # Running on Hugging Face Spaces
        args.host = "0.0.0.0"
        args.port = 7860
        args.share = False
        inbrowser = False
    else:
        inbrowser = True
    
    logger.info(f"Starting DNA-Diffusion Gradio app on {args.host}:{args.port}")
    
    demo.launch(
        share=args.share,
        server_name=args.host,
        server_port=args.port,
        inbrowser=inbrowser
    )