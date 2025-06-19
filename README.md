---
license: apache-2.0
---
## Vision Transformer (ViT) Models for Digital Forensics

This repository provides Vision Transformer (ViT) models fine-tuned to detect manipulated (fake) versus authentic (real) image frames extracted from the FaceForensics++ dataset. The models were trained using Intel® ARC GPU (XPU-enabled) and optimized for binary image classification in digital forensics workflows.

---

## 🧠 Models Trained

| Model Name                 | Pretrained On         | Patch Size | Parameters |
|---------------------------|-----------------------|------------|------------|
| `vit_tiny_patch16_224`    | ImageNet-21k          | 16×16      | ~5.7M      |
| `vit_tiny_patch32_224`    | ImageNet-21k          | 32×32      | ~5.6M      |
| `vit_small_patch32_224`   | AugReg + IN21k + IN1k | 32×32      | ~22M       |
| `vit_large_patch16_224`   | ImageNet-21k          | 16×16      | ~304M      |
| `vit_large_patch32_224`   | ImageNet-21k          | 32×32      | ~304M      |

---

## 🗂️ Dataset

- **Name**: DeepFake Detection (DFD)  
- **Source**: [Kaggle DFD Dataset] 
- **Classes**: `real`, `fake`  
- **Input**: Extracted video frames resized to 224×224 RGB images  
- **Preprocessing**:
  - Resizing and normalization using `torchvision.transforms`
  - Structured into `train/real`, `train/fake`, `val/real`, `val/fake`

---

## ⚙️ Hardware & Environment

- **Accelerator**: Intel® ARC GPU (XPU via Intel Extension for PyTorch)
- **Frameworks**:
  - PyTorch 2.7.0 + XPU backend
  - torchvision 0.22.0
  - timm for pretrained ViT models
- **OS**: Windows 11
- **Memory Consideration**: `vit_huge_patch14_224` requires large GPU memory; tested on Intel ARC A770 16GB and NPU Boost

---

## ✅ Use Case: Deepfake Frame Detection

These models are designed to identify manipulated media content at the frame level. Use cases include:

- 🔍 Video forensics
- 🎞️ Deepfake screening and flagging pipelines
- 🧪 Data validation for machine learning datasets
- 📡 Real-time frame-level media authentication

They are well-suited for deployment in digital forensics, content moderation, and research scenarios where image authenticity is critical.

---

## 📊 Results 

| Model | Train Accuracy | Validation Accuracy |
|-------|----------------|---------------------|
| [vit_large_patch16_224](https://huggingface.co/pranav2711/VisionTransformerDigitalForensics/blob/main/vit_large_patch16_224.pth) | **94.89%** | **91.22%** |
| [vit_large_patch32_224](https://huggingface.co/pranav2711/VisionTransformerDigitalForensics/blob/main/vit_large_patch32_224.pth) | 91.31% | 89.23% |
| [vit_tiny_patch16_224](https://huggingface.co/pranav2711/VisionTransformerDigitalForensics/blob/main/vit_tiny_patch16_224.pth) | 92.41% | 89.20% |
| [vit_small_patch32_224](https://huggingface.co/pranav2711/VisionTransformerDigitalForensics/blob/main/vit_small_patch32_224.pth) | 91.38% | 88.29% |
| [vit_small_patch16_224](https://huggingface.co/pranav2711/VisionTransformerDigitalForensics/blob/main/vit_small_patch32_224.pth) | 80.67% | 81.25% |
| [vit_base_patch16_224](http://huggingface.co/pranav2711/VisionTransformerDigitalForensics/blob/main/vit_base_patch16_224.pth) | 90.65% | 85.36% |
| [vit_base_patch32_224](https://huggingface.co/pranav2711/VisionTransformerDigitalForensics/blob/main/vit_base_patch32_224.pth) | 79.54% | 79.54% |

**vit_large_patch16_224**
This model achieved the highest validation accuracy of 91.22% with strong training stability and generalization. It is recommended as the final model for deployment or downstream tasks.

---

## 📄 License

This model is licensed under the [CreativeML OpenRAIL-M License](https://huggingface.co/spaces/CompVis/stable-diffusion-license).  
It allows for responsible research and commercial use, but **strictly prohibits**:

- Harassment, surveillance, or profiling.
- Generating misleading or harmful content (e.g., deepfakes for impersonation).
- Use in political campaigns or autonomous weapons.

Please read the license carefully before using the model.
 
