from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
import torch
import sys

def classify_with_model(model_name, model_display_name, device):
    try:
        results = []
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
        text_labels = ["a photo of people", "scooter or bike"]

        image_dir = Path("data")
        if not image_dir.exists():
            print(f"error: directory '{image_dir}' not found")
            return None, None
        image_files = sorted(list(image_dir.glob("*.jpg"))) #binary naming??
        if len(image_files) == 0:
            print(f"error no files '{image_dir}'")
            return None, None

        #process images
        for img_path in image_files:
            try:
                image = Image.open(img_path).convert('RGB')
                inputs = processor(text=text_labels, images=image, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                    prediction = probs.argmax().item()
                results.append(prediction)
                    
            except Exception as e:
                print(f"error for {img_path.name}: {e}")
                results.append(0)
        
        #generate qr code
        results_arr = np.array(results)
        size = int(np.sqrt(len(results_arr)))
        if size * size < len(results_arr):
            size += 1
            padded_results = np.zeros(size * size)
            padded_results[:len(results_arr)] = results_arr
            results_arr = padded_results
        
        qr_image = 1 - results_arr.reshape((size, size))
        qr_pil = Image.fromarray((qr_image * 255).astype(np.uint8), mode='L')
        qr_pil_large = qr_pil.resize((qr_pil.width * 20, qr_pil.height * 20), Image.NEAREST)
        safe_name = model_display_name.replace("/", "_").replace(" ", "_")
        filename = f'qr_code_{safe_name}.png'
        qr_pil_large.save(filename)  
        
        del model #clean
        torch.cuda.empty_cache() #clean
        return qr_image, filename
        
    except Exception as e:
        print(f"error with {model_display_name}: {e}")
        return None, None

def main():
    results_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_to_test = [("openai/clip-vit-base-patch32", "CLIP_ViT-B-32"),
        ("openai/clip-vit-base-patch16", "CLIP_ViT-B-16"),
        ("openai/clip-vit-large-patch14", "CLIP_ViT-L-14"),]
    
    for model_name, display_name in models_to_test:
        qr_image, filename = classify_with_model(model_name, display_name, device)
        results_dict[display_name] = (qr_image, filename)
    
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    for idx, (name, (qr_img, fname)) in enumerate(results_dict.items()):
        axes[idx].imshow(qr_img, cmap='gray')
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    plt.tight_layout()
    plt.switch_backend('Agg')
    plt.savefig('qrcodecomparison.png', dpi=200, bbox_inches='tight')
    print("\ngenerated qr codes:")
    for name, (_, fname) in results_dict.items():
        print(f"  {fname} ({name})")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"fatal error: {e}")
        sys.exit(1)