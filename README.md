# Learning to Pick and Place: End-to-End Imitation Learning with a Moving Target
This project teaches a robot how to pick up an object and place it into a moving box — just by watching demonstrations.
Instead of being programmed step-by-step, the robot learns an end-to-end policy that maps what it sees to the actions it should take.

The goal is to make the robot adapt to different positions of the object and box, so it can perform the task smoothly even when things move between runs.

# Demo
https://github.com/user-attachments/assets/58971d33-b68e-4dfc-94f5-259e81449268

### 2. System Description
### 2.1 Task and Setup

The setup involves a 5-DOF robotic arm (LeRobot SO-100) [2].
LeRobot is used only for low-level control—to send joint position commands and read robot feedback.
All other components, including data recording, camera capture, trajectory saving, training, and replay, were implemented from scratch for this project.

The task is a pick-and-place motion: the robot must pick an object from the table and place it into a box.
In each run, both the object and the box start at different positions.
The robot receives only the camera image and its own joint positions; it has no prior knowledge of where the targets are.

---

### 2.2 Learning Objective

The policy is trained end-to-end to map recent observations to a short sequence of future joint positions, referred to as an action chunk.
Each action chunk represents the upcoming segment of motion in joint space, enabling the model to plan several steps ahead instead of producing single-step commands.

The approach is inspired by the Action Chunking Transformer (ACT) [1] but differs in key design choices.
Unlike ACT, this model does not use latent variables **z** and employs a **frozen SigLIP** image encoder for visual representation.

Each training sample includes:

Observation: images from both the wrist-mounted and third-person cameras, together with the robot joint-state vector.

---

### 2.3 Model Architecture

The policy follows a **Transformer encoder–decoder** design that processes multi-camera image observations and robot joint states to predict a sequence of future joint positions.


#### ⚙️ Encoder

Each observation includes synchronized images from two cameras (wrist-mounted and third-person) and the robot joint-state vector.

**1. Image processing**
  - Each image is passed through a frozen **SigLIP** encoder, producing feature maps of shape:
    ` [B, Np, Dimg] `
  - where:
      - **B** — batch size  
      - **Np** — number of image patches per camera  
      - **Dimg** — image feature dimension  
  - Features from all cameras are concatenated along the patch dimension:
    ` [B, Nc * Np, Dimg] `
  - where **Nc** is the number of cameras.

**2. Projection and tokenization**
  - Each image token is linearly projected to the model dimension `Dmodel`.  
  - A learnable **image-class token** is added to each projected embedding:
    ` [B, Nc * Np, Dmodel] `

**3. Robot-state embedding**
  - The (5-DOF joint-position + gripper state) vector is linearly projected into the same latent space and prepended  
    with a learnable **state-class token**:
    ` [B, 1, Dmodel] `

**4. Encoder input**
  - Image and state tokens are concatenated to form:
    ` [B, Nc * Np + 1, Dmodel] `
  - This sequence is processed by several Transformer encoder blocks using  
    **self-attention (no masking)** to integrate spatial and state information across all tokens.

#### ⚙️ Decoder

The decoder predicts a chunk of `k` future joint positions.

**1. Input tokens**
  - A set of `k` **zero tokens** is initialized, each augmented with **positional embeddings** to encode temporal order.
  - After projection to `Dmodel`, the decoder input becomes:
    ` [B, k, Dmodel] `

**2. Cross-attention and output**
  - The decoder performs **cross-attention** with the encoder outputs to condition  
    the action tokens on the current scene.  
  - A final **MLP head** maps each token to the joint-space output:
    ` [B, k, ActionDim] `
  - where ActionDim = 6; in robotic terms, the arm itself is 5-DoF, but the action model includes an additional dimension for the gripper stat

---

### 2.4 Training Setup

The model was trained end-to-end using an **L2 loss** between predicted and ground-truth joint positions.  
Training ran for **57 epochs**, using **100 recorded demonstrations**, which resulted in **50,450 training samples** after chunking and sequence preparation.  
A **5% validation split** was reserved to monitor performance and prevent overfitting during training.

#### ⚙️ Model and Hyperparameters

  - **Model objective:** L2 loss (mean squared error on joint positions)  
  - **Training data:** 100 demonstrations → 50,450 samples  
  - **Epochs:** 57  
  - **Validation:** 5% of data reserved for validation  

  - **Model configuration:**
      - `d_model`: 512  
      - `d_internal`: 1024  
      - `h`: 8  
      - `dropout`: 0.1  
      - `encoder_num`: 4  
      - `decoder_num`: 7  
      - `prediction_horizon`: 100  
      - `actions_dim`: 6  
      - `img_features`: 1152  
      - `H_patches × W_patches`: 16 × 16  
      - `cam_keys`: ['third_person_view', 'wrist_view']  

  - **Training configuration:**
      - `batch_size`: 64  
      - `lr`: 1e-5  
      - `beta1`: 0.9  
      - `beta2`: 0.98  
      - `eps`: 1e-9  

#### 🧠 Optimization

  - Optimizer: **Adam**  
  - Early stopping and validation monitoring helped prevent overfitting

---

### References

[1] Zhao, T. Z. et al. **"Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware."** (2023).  
[https://arxiv.org/abs/2304.13705](https://arxiv.org/abs/2304.13705)


[2] **LeRobot**: Open-source imitation learning framework for robotics (2024).  
[https://github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)





