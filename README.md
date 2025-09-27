Here is the wrokflow of DunHuang Mural Restoration process:
![alt text](https://github.com/rili0214/Dunhuang/blob/main/Images/Workflow.png)


# MuralLoss: Combined Loss for Mural Restoration

This project implements a custom loss function **MuralLoss**, designed for mural restoration tasks.  
It combines multiple complementary components to balance pixel-level accuracy, structural fidelity, perceptual similarity, and fine details.

## 📌 Loss Components

### 1. L1 Loss
- Ensures pixel-wise accuracy (color and brightness consistency).  
- Widely used as a baseline in image restoration.  

### 2. SSIM Loss
- Measures **structural similarity** between restored and ground-truth images.  
- More consistent with human visual perception compared to L1/MSE.  
- Formula:  
  ![SSIM Formula](https://latex.codecogs.com/png.latex?SSIM(x,y)=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)})  

  (Plain text version for GitHub preview:)  
  `SSIM(x,y) = ((2 * μx * μy + C1) * (2 * σxy + C2)) / ((μx^2 + μy^2 + C1) * (σx^2 + σy^2 + C2))`


- Reference:  
  > Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).  
  > *Image quality assessment: From error visibility to structural similarity*.  
  > IEEE Transactions on Image Processing, 13(4), 600–612.

### 3. Perceptual Loss
- Uses deep feature maps (e.g., VGG) to measure semantic and perceptual similarity.  
- Encourages the restored mural to preserve high-level content and style.  
- Reference:  
  > Johnson, J., Alahi, A., & Fei-Fei, L. (2016).  
  > *Perceptual losses for real-time style transfer and super-resolution*.  
  > ECCV.

### 4. Edge-aware Loss
- Extracts edges via Sobel filters and minimizes their difference.  
- Ensures sharper edges, textures, and cracks — critical for mural restoration.  
- Reference:  
  > Zhang, K., Zuo, W., & Zhang, L. (2019).  
  > *Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels*.  
  > CVPR.

---

## ⚖️ Total Loss

The final loss is a weighted combination:

![Total Loss](https://latex.codecogs.com/png.latex?\mathcal{L}_{total}=\lambda_1\mathcal{L}_{L1}+\lambda_2\mathcal{L}_{SSIM}+\lambda_3\mathcal{L}_{Perceptual}+\lambda_4\mathcal{L}_{Edge})

(Plain text version:)  
`L_total = λ1 * L1 + λ2 * L_SSIM + λ3 * L_Perceptual + λ4 * L_Edge`


Default weights:
- `l1_weight = 1.0`  
- `ssim_weight = 0.5`  
- `perceptual_weight = 0.1`  
- `edge_weight = 0.2`

---

## 🖼️ Why MuralLoss?

- **L1**: keeps global colors accurate  
- **SSIM**: preserves structure  
- **Perceptual**: maintains semantic meaning  
- **Edge-aware**: enhances fine details  
This makes it especially suitable for **mural restoration**, where both **visual fidelity** and **structural consistency** are crucial.

## Phase 1: Freeze Encoder, Train Decoder Only

### Training Setup
- **Goal**: Fine-tune the decoder on the Mural dataset while keeping the encoder frozen (feature extractor remains fixed).
- **Configuration**:
  - Optimizer: **Adam** with cosine annealing learning rate scheduler
  - Loss: Weighted combination of **L1 + SSIM + Perceptual Loss**
  - Batch size: **8**
  - Image size: **256 × 256**
  - Initial learning rate: **1e-4**
  - Epochs: **50** (with early stopping, patience = 5)

### Training Process
- The encoder parameters were frozen during this stage.
- Only the decoder parameters were updated.
- Both training and validation metrics (Loss, PSNR) were monitored with TensorBoard.

### Results
- Training stopped at **epoch 40** due to early stopping.
- **Best model performance**:
  - Train Loss ≈ **0.22**
  - Validation Loss ≈ **0.24**
  - Train PSNR ≈ **23.1 dB**
  - Validation PSNR ≈ **21.5 dB**

### Visualization
Training and validation curves:

![Phase1 Loss](images/phase1.png)

### Conclusion
- The model shows clear convergence: training and validation losses decrease steadily, while PSNR increases.
- Validation PSNR reached **21.5 dB**, showing noticeable improvement over the initial state.
- This phase successfully adapted the decoder. The next step will be **Phase 2: Unfreeze encoder and fine-tune the whole model**.







