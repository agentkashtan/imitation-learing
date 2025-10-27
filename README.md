# Learning to Pick and Place: End-to-End Imitation Learning with a Moving Target
This project teaches a robot how to pick up an object and place it into a moving box — just by watching demonstrations.
Instead of being programmed step-by-step, the robot learns an end-to-end policy that maps what it sees to the actions it should take.

The goal is to make the robot adapt to different positions of the object and box, so it can perform the task smoothly even when things move between runs.

# Demo
https://github.com/user-attachments/assets/58971d33-b68e-4dfc-94f5-259e81449268

#2. System Description
2.1 Task and Setup

The setup involves a 5-DOF robotic arm (LeRobot SO-100).
LeRobot is used only for low-level control—to send joint position commands and read robot feedback.
All other components, including data recording, camera capture, trajectory saving, training, and replay, were implemented from scratch for this project.

The task is a pick-and-place motion: the robot must pick an object from the table and place it into a box.
In each run, both the object and the box start at different positions.
The robot receives only the camera image and its own joint positions; it has no prior knowledge of where the targets are.

2.2 Learning Objective

The policy is trained end-to-end to map recent observations to a short sequence of future joint positions, referred to as an action chunk.
Each action chunk represents the upcoming segment of motion in joint space, enabling the model to plan several steps ahead instead of producing single-step commands.

The approach is inspired by the Action Chunking Transformer (ACT) but differs in key design choices.
Unlike ACT, this model does not use latent variables 
𝑧
z and employs a frozen SigLIP image encoder for visual representation.
This makes the policy fully deterministic and simpler to train.

Each training sample includes:

Observation: images from both the wrist-mounted and third-person cameras, together with the robot joint-state vector.

Action chunk: a sequence of future joint positions representing the intended motion.

The policy learns to generate these motion chunks directly from recent visual and proprioceptive input, forming a smooth, task-adaptive trajectory.
