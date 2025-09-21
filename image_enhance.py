import os
import multiprocessing

# Force CPU-only operation
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())

import io
import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify
from PIL import Image, ImageEnhance , ImageFilter
import urllib.request
import gdown

app = Flask(__name__)

class RealESRGANEnhancer:
    def __init__(self):
        """
        Initialize Real-ESRGAN model for image enhancement
        """
        self.model = None
        self.setup_model()
    
    def setup_model(self):
        """
        Setup Real-ESRGAN model - downloads if needed
        """
        print("Setting up Real-ESRGAN model...")
        
        try:
            # Try to import realesrgan
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            # Model paths
            model_path = 'weights/RealESRGAN_x4plus.pth'
            os.makedirs('weights', exist_ok=True)
            
            # Download model if it doesn't exist
            if not os.path.exists(model_path):
                print("Downloading Real-ESRGAN model...")
                model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
                urllib.request.urlretrieve(model_url, model_path)
                print("Model downloaded successfully!")
            
            # Initialize model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            
            self.model = RealESRGANer(
                scale=4,
                model_path=model_path,
                dni_weight=None,
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False,
                gpu_id=None  # Use CPU
            )
            print("Real-ESRGAN model loaded successfully on CPU!")
            
        except ImportError:
            print("Real-ESRGAN not installed. Install with: pip install realesrgan")
            self.model = None
        except Exception as e:
            print(f"Error setting up Real-ESRGAN: {e}")
            self.model = None
    
    def enhance_image(self, image, scale=2):
        """
        Enhance image using Real-ESRGAN
        """
        if self.model is None:
            # Fallback to traditional enhancement if Real-ESRGAN not available
            return self.fallback_enhancement(image, scale)
        
        try:
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # Enhance using Real-ESRGAN
            print(f"Enhancing with Real-ESRGAN (scale: {scale}x)...")
            enhanced_array, _ = self.model.enhance(img_array, outscale=scale)
            
            # Convert back to PIL
            enhanced_image = Image.fromarray(enhanced_array)
            
            return enhanced_image
            
        except Exception as e:
            print(f"Real-ESRGAN enhancement failed: {e}")
            return self.fallback_enhancement(image, scale)
    
    def fallback_enhancement(self, image, scale=2):
        """
        Fallback enhancement method if Real-ESRGAN fails
        """
        print("Using fallback enhancement method...")
        
        # Resize image
        new_size = (int(image.width * scale), int(image.height * scale))
        enhanced = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Apply traditional enhancements
        # Noise reduction using PIL
        enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))
        
        # Enhance sharpness
        sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = sharpness_enhancer.enhance(1.2)
        
        # Enhance color
        color_enhancer = ImageEnhance.Color(enhanced)
        enhanced = color_enhancer.enhance(1.15)
        
        # Enhance contrast
        contrast_enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = contrast_enhancer.enhance(1.1)
        
        # Enhance brightness slightly
        brightness_enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = brightness_enhancer.enhance(1.05)
        
        return enhanced

# Initialize the enhancer
print("Initializing Real-ESRGAN Enhancer...")
enhancer = RealESRGANEnhancer()

@app.route('/enhance-image', methods=['POST'])
def enhance_image():
    """
    Enhance image using Real-ESRGAN for e-commerce appeal
    """
    # Input validation
    if 'image' not in request.files:
        return jsonify({"error": "No 'image' file part in the request"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected for uploading"}), 400
    
    # Get enhancement parameters
    scale = int(request.form.get('scale', 2))  # 1, 2, 3, or 4
    quality = request.form.get('quality', 'high')  # low, medium, high
    
    # Validate scale
    if scale not in [1, 2, 3, 4]:
        scale = 2
    
    try:
        # Load and validate image
        input_image = Image.open(file.stream).convert("RGB")
        print(f"Processing image: {input_image.size} with {scale}x enhancement")
        
        # Limit input size to prevent memory issues
        max_input_size = 2000
        if max(input_image.size) > max_input_size:
            print(f"Resizing large input image from {input_image.size}")
            input_image.thumbnail((max_input_size, max_input_size), Image.Resampling.LANCZOS)
            print(f"Resized input to: {input_image.size}")
            
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {e}"}), 400
    
    print("Starting Real-ESRGAN enhancement...")
    
    try:
        # Enhance using Real-ESRGAN
        enhanced_image = enhancer.enhance_image(input_image, scale=scale)
        
        print(f"Enhancement complete! Output size: {enhanced_image.size}")
        
        # Additional post-processing based on quality setting
        if quality == 'high':
            print("Applying high-quality post-processing...")
            # Fine-tune the enhanced image
            contrast_enhancer = ImageEnhance.Contrast(enhanced_image)
            enhanced_image = contrast_enhancer.enhance(1.05)
            
            sharpness_enhancer = ImageEnhance.Sharpness(enhanced_image)
            enhanced_image = sharpness_enhancer.enhance(1.1)
        
        # Save locally in script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_name = file.filename.split('.')[0] if file.filename else 'image'
        output_filename = f"{base_name}_enhanced_{scale}x_{quality}.jpg"
        output_path = os.path.join(script_dir, output_filename)
        
        # Save with high quality
        enhanced_image.save(output_path, 'JPEG', quality=95, optimize=True)
        print(f"Enhanced image saved to: {output_path}")
        
        # Prepare response
        byte_arr = io.BytesIO()
        enhanced_image.save(byte_arr, format='JPEG', quality=95, optimize=True)
        byte_arr.seek(0)
        
        return send_file(
            byte_arr,
            mimetype='image/jpeg',
            
            
        )
        
    except Exception as e:
        print(f"Enhancement error: {e}")
        return jsonify({"error": f"Image enhancement failed: {e}"}), 500

@app.route('/batch-enhance', methods=['POST'])
def batch_enhance():
    """
    Enhance multiple images at once
    """
    if 'images' not in request.files:
        return jsonify({"error": "No 'images' files in the request"}), 400
    
    files = request.files.getlist('images')
    scale = int(request.form.get('scale', 2))
    quality = request.form.get('quality', 'medium')
    
    if not files:
        return jsonify({"error": "No images selected"}), 400
    
    results = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for file in files:
        try:
            input_image = Image.open(file.stream).convert("RGB")
            enhanced_image = enhancer.enhance_image(input_image, scale=scale)
            
            base_name = file.filename.split('.')[0] if file.filename else f'image_{len(results)+1}'
            output_filename = f"{base_name}_enhanced_{scale}x.jpg"
            output_path = os.path.join(script_dir, output_filename)
            
            enhanced_image.save(output_path, 'JPEG', quality=95, optimize=True)
            
            results.append({
                "original": file.filename,
                "enhanced": output_filename,
                "status": "success"
            })
            
        except Exception as e:
            results.append({
                "original": file.filename,
                "enhanced": None,
                "status": "failed",
                "error": str(e)
            })
    
    return jsonify({
        "message": f"Processed {len(files)} images",
        "results": results
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Real-ESRGAN image enhancer is running",
        "model_loaded": enhancer.model is not None,
        "available_scales": [1, 2, 3, 4],
        "available_qualities": ["low", "medium", "high"]
    })

@app.route('/compare-scales', methods=['POST'])
def compare_scales():
    """
    Generate images at different scales for comparison
    """
    if 'image' not in request.files:
        return jsonify({"error": "No 'image' file part in the request"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected for uploading"}), 400
    
    try:
        input_image = Image.open(file.stream).convert("RGB")
        
        scales = [1, 2, 3, 4]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_name = file.filename.split('.')[0] if file.filename else 'image'
        results = {}
        
        for scale in scales:
            print(f"Generating {scale}x version...")
            enhanced = enhancer.enhance_image(input_image, scale=scale)
            
            output_filename = f"{base_name}_{scale}x_enhanced.jpg"
            output_path = os.path.join(script_dir, output_filename)
            
            enhanced.save(output_path, 'JPEG', quality=95, optimize=True)
            results[f"{scale}x"] = {
                "filename": output_filename,
                "size": enhanced.size
            }
        
        return jsonify({
            "message": "All scale versions created successfully",
            "original_size": input_image.size,
            "enhanced_versions": results
        })
        
    except Exception as e:
        return jsonify({"error": f"Scale comparison failed: {e}"}), 500

if __name__ == '__main__':
    print("=" * 70)
    print("Real-ESRGAN E-commerce Image Enhancer")
    print("=" * 70)
    print("Features:")
    print("- Real-ESRGAN super-resolution (1x to 4x scaling)")
    print("- Intelligent detail enhancement")
    print("- Professional image upscaling")
    print("- Batch processing support")
    print("- Fallback enhancement if Real-ESRGAN unavailable")
    print("=" * 70)
    print("Installation requirements:")
    print("pip install realesrgan basicsr")
    print("=" * 70)
    print("Usage examples:")
    print("curl -X POST -F 'image=@image.jpg' -F 'scale=2' http://localhost:9000/enhance-image --output enhanced.jpg")
    print("curl -X POST -F 'image=@image.jpg' http://localhost:9000/compare-scales")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=9000, debug=True)