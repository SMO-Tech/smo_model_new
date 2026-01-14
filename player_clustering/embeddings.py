import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import supervision as sv
from transformers import AutoProcessor, SiglipVisionModel
import torch
import numpy as np
from tqdm import tqdm
from more_itertools import chunked
from PIL import Image
import cv2

# Import GPU settings from constants
try:
    from constants import USE_GPU
except ImportError:
    USE_GPU = True

def get_device():
    """Get the appropriate device for inference, respecting USE_GPU setting."""
    if not USE_GPU:
        return 'cpu'
    
    if not torch.cuda.is_available():
        return 'cpu'
    
    try:
        # Test GPU compatibility
        test_tensor = torch.zeros(1).cuda()
        _ = test_tensor * 2
        del test_tensor
        torch.cuda.empty_cache()
        return 'cuda'
    except (RuntimeError, Exception) as e:
        if "no kernel image" in str(e) or "CUDA capability" in str(e):
            return 'cpu'
        raise

device = get_device()


class EmbeddingExtractor:
    """
    Handles extraction of visual embeddings from player crops using SigLIP model.
    """
    
    def __init__(self, model_name="google/siglip-base-patch16-224"):
        """
        Initialize the embedding extractor with SigLIP model.
        
        Args:
            model_name: HuggingFace model name for SigLIP
        """
        current_device = get_device()
        self.model = SiglipVisionModel.from_pretrained(model_name).to(current_device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Verify model is on correct device
        try:
            if current_device != 'cpu':
                # Test if model can run on GPU
                test_input = torch.zeros(1, 3, 224, 224).to(current_device)
                _ = self.model.vision_model.embeddings.patch_embedding(test_input)
                del test_input
                torch.cuda.empty_cache()
        except RuntimeError as e:
            if "no kernel image" in str(e) or "CUDA capability" in str(e):
                print(f"⚠️  SigLIP model incompatible with GPU, moving to CPU")
                self.model = self.model.cpu()
            else:
                raise
        
    def get_player_crops(self, frame, player_detections):
        """
        Extract player crops from frame using detection bounding boxes.
        
        Args:
            frame: Input video frame
            player_detections: Player detection results
            
        Returns:
            List of cropped player images in PIL format
        """
        cropped_images = []
        for boxes in player_detections.xyxy:
            cropped_image = sv.crop_image(frame, boxes)
            # Skip empty crops (invalid bounding boxes)
            if cropped_image is not None and cropped_image.size > 0:
                cropped_images.append(cropped_image)
        
        # Convert OpenCV BGR to PIL RGB format
        # Use direct conversion instead of sv.cv2_to_pillow for compatibility
        pil_images = []
        for cropped_image in cropped_images:
            # Check if image is valid and not empty
            if cropped_image is None or cropped_image.size == 0:
                continue
            try:
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(rgb_image)
                pil_images.append(pil_image)
            except cv2.error as e:
                # Skip invalid crops (empty or malformed images)
                continue
        return pil_images
    
    def create_batches(self, data, batch_size=24):
        """
        Create batches from data for efficient processing.
        
        Args:
            data: Input data to batch
            batch_size: Size of each batch
            
        Returns:
            List of batched data
        """
        return list(chunked(data, batch_size))
    
    def get_embeddings(self, image_batches):
        """
        Extract SigLIP embeddings from image batches.
        
        Args:
            image_batches: Batched images for processing
            
        Returns:
            Numpy array of embeddings
        """
        data = []
        current_device = get_device()  # Get current device (may have changed)
        total_batches = len(image_batches)
        total_images = sum(len(batch) for batch in image_batches)
        
        # Only show progress for large batches (training) to avoid spam during frame processing
        if total_images > 100:
            print(f"Processing {total_batches} batches ({total_images} images) of embeddings on {current_device}...")
        
        try:
            with torch.no_grad():
                # Use tqdm only for large batches
                batch_iter = tqdm(image_batches, desc='extracting_embeddings', total=total_batches) if total_images > 100 else image_batches
                for batch_idx, batch in enumerate(batch_iter):
                    try:
                        if total_images > 100 and (batch_idx % 10 == 0 or batch_idx == 0):
                            print(f"  Batch {batch_idx+1}/{total_batches} on {current_device}")
                        inputs = self.processor(images=batch, return_tensors="pt").to(current_device)
                        outputs = self.model(**inputs)
                        embeddings = torch.mean(outputs.last_hidden_state, dim=1)
                        # Keep on GPU if using cuML (GPU-accelerated clustering), otherwise move to CPU
                        if current_device != 'cpu':
                            # Keep as tensor on GPU - will convert to numpy only when needed
                            data.append(embeddings)
                        else:
                            data.append(embeddings.cpu().numpy())
                        
                        # Clear GPU cache periodically
                        if batch_idx % 20 == 0 and current_device != 'cpu':
                            torch.cuda.empty_cache()
                            
                    except RuntimeError as e:
                        error_msg = str(e)
                        if "no kernel image" in error_msg or "CUDA capability" in error_msg:
                            # Fallback to CPU
                            print(f"⚠️  GPU operation failed at batch {batch_idx+1}, retrying with CPU")
                            current_device = 'cpu'
                            # Move model to CPU if not already
                            if next(self.model.parameters()).is_cuda:
                                self.model = self.model.cpu()
                            inputs = self.processor(images=batch, return_tensors="pt").to('cpu')
                            outputs = self.model(**inputs)
                            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                            data.append(embeddings)
                        elif "out of memory" in error_msg.lower():
                            print(f"⚠️  GPU out of memory at batch {batch_idx+1}, reducing batch size or using CPU")
                            # Try with smaller batch or CPU
                            current_device = 'cpu'
                            if next(self.model.parameters()).is_cuda:
                                self.model = self.model.cpu()
                            inputs = self.processor(images=batch, return_tensors="pt").to('cpu')
                            outputs = self.model(**inputs)
                            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                            data.append(embeddings)
                        else:
                            print(f"❌ Error at batch {batch_idx+1}: {error_msg}")
                            raise
                    except Exception as e:
                        print(f"❌ Unexpected error at batch {batch_idx+1}: {type(e).__name__}: {str(e)}")
                        raise
                        
            if len(data) == 0:
                raise ValueError("No embeddings extracted from batches")
            
            # Convert tensors to numpy if needed (for CPU or if cuML not available)
            if isinstance(data[0], torch.Tensor):
                # All are tensors - concatenate on GPU then move to CPU
                data = torch.cat(data, dim=0).cpu().numpy()
            else:
                # All are numpy arrays - concatenate normally
                data = np.concatenate(data, axis=0)
            
            # Only print for large batches to avoid spam during frame processing
            if len(data) > 100:
                print(f"✅ Successfully extracted {len(data)} embeddings")
            import sys
            sys.stdout.flush()
            return data
        except KeyboardInterrupt:
            print("\n⚠️  Embedding extraction interrupted by user")
            import sys
            sys.stdout.flush()
            if len(data) > 0:
                print(f"Returning {len(data)} embeddings collected so far...")
                return np.concatenate(data, axis=0)
            raise
        except Exception as e:
            print(f"❌ Fatal error during embedding extraction: {type(e).__name__}: {str(e)}")
            import sys
            sys.stdout.flush()
            raise