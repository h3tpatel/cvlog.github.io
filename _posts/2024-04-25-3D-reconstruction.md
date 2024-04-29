# Unconstrained image-based dense 3D reconstruction

My primary objective for this project is to learn and implement a method for 3D reconstruction from a limited set of arbitrary image collections, without requiring prior information on camera calibration or even viewpoint poses. In this approach, I aim to overcome the limitations of traditional methods like Structure-from-Motion (SfM) and NeRF, which typically require extensive and precise input data.

### Experimentation and it's Implementation:

<iframe src="https://tinyglb.com/viewer/ebaa784058cf48e6a16a749f8bf46d66" style="border: 0; height: 600px; width: 100%"></iframe>

**embedded as a pointcloud:**

<iframe src="https://tinyglb.com/viewer/188387d9d50c482993061a75ab821888" style="border: 0; height: 600px; width: 100%"></iframe>

**visualization of rgb, depth, confidence metrics...**

![rgb, depth, confidence](https://github.com/h3tpatel/cvlog.github.io/assets/144167031/3c282c17-dedc-4ca6-a10f-db37c6dfb020)

```text
>> **Note:** inference is performed on the CPU.
>> Inference with model on 6 image pairs
100%|████████████████████████████████████████████████████████████████| 6/6 [00:34<00:00,  5.68s/it]
 init edge (1*,0*) score=21.2039737701416
 init edge (2*,0) score=14.447808265686035
 init loss = 0.004692849703133106
Global alignement - optimizing for:
['pw_poses', 'im_depthmaps', 'im_poses', 'im_focals', 'im_conf.0', 'im_conf.1', 'im_conf.2']
100%|█████████████████████████████████| 1000/1000 [01:19<00:00, 12.59it/s, lr=0.01 loss=0.00170984]
(exporting 3D scene to /var/folders/y5/v82tll4j1n1b7_pn428d3s880000gn/T/scene.glb)
```

#### Challenges with Structure-from-Motion (SfM):

![Image Matching](https://production-media.paperswithcode.com/tasks/image_matching_4Ht6D90.png)

**Dependency on Feature Matching**: relies heavily on detecting and matching features across multiple images. For example, in scenes with low texture or repetitive patterns, or with slender objects, feature matching can be unreliable, leading to poor reconstruction.


![](https://demuc.de/colmap/sparse-reconstruction.png)

**Structure from Motion (SfM)**: requires sufficient and diverse camera motion to triangulate points accurately. For instance, in scenarios with limited or linear camera motion, SfM can fail to resolve depth accurately, resulting in flat or distorted reconstructions.

**Computational Complexity:** SfM involves solving complex optimization problems to determine camera parameters and 3D point locations, which can be computationally intense. Like, large-scale reconstructions require significant computational resources and time, especially when adjusting parameters through bundle adjustment.

### Overview of 3D Reconstruction: Its Pipeline and Methodologies:

![overview](https://github.com/h3tpatel/cvlog.github.io/assets/144167031/b3d6b2cc-3af3-42d2-b97b-d7f694f99892)


| Pipeline Stage                                           | Meaning                                                                                                                                                                                       |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Input Acquisition                                        | A collection of images with unknown camera poses and intrinsic parameters.                                                                                                                    |
| Network Architecture                                     | Images processed through a Vision Transformer (ViT) encoder, with dual transformer decoders extracting geometric relationships and regression heads outputting pointmaps and confidence maps. |
| Pointmap Regression                                      | Network regresses dense 2D fields of 3D points (pointmaps) directly from pairs of images.                                                                                                     |
| Confidence Maps                                          | Outputs confidence maps alongside pointmaps to assess the reliability of the predicted 3D positions.                                                                                          |
| Training Objective                                       | Uses a regression loss based on the Euclidean distance in 3D space, adjusted by confidence scores to prioritize reliable predictions.                                                         |
| Global Alignment for Multi-View Reconstruction           | Aligns pointmaps from multiple views into a common 3D space, optimizing alignment directly in 3D space.                                                                                       |
| Extraction of Geometric Properties and Camera Parameters | Derives depth information, pixel correspondences, and relative camera poses from pointmaps, and infers camera parameters from spatial relationships.                                          |
| Output                                                   | Produces a 3D model of the scene, applicable in virtual reality, gaming, and urban planning, among others.                                                                                    |

### Architecture of Unconstrained image-based dense 3D reconstruction:

<iframe src="https://tinyglb.com/viewer/139565c2268f4f86b92206daf60ee5a9" style="border: 0; height: 600px; width: 100%"></iframe>

![architecture](https://github.com/h3tpatel/cvlog.github.io/assets/144167031/50bf0f49-018b-4168-afcc-dfdc6af17784)


- **How is a point represented in 3D using a pointmap?**: a pointmap is a dense 2D array of 3D points, expressed as $$X \in \mathbb{R}^{W \times H \times 3}$$, where $$W$$ and $$H$$ denote the width and height of the image, respectively.

- This structure allows for a one-to-one mapping between image pixels and 3D scene points, such that each pixel $$(i, j)$$ in the image corresponds to a 3D point $$X_{i,j}$$ in the scene, represented as $$I_{i,j} \leftrightarrow X_{i,j}$$ for all pixel coordinates $$(i, j)$$.

- This simplifies the process of extracting geometric data, such as depth and camera parameters, as it allows for a direct mapping from image pixels to their corresponding 3D points, enhancing tasks like multi-view stereo reconstruction without explicit knowledge of camera parameters.

- **Siamese Network with Shared Weights**: Two branches with shared Vision Transformer (ViT) encoders for feature extraction from input image pairs.

- **Transformer Decoders**: Two decoders process features with cross-attention to ensure output pointmaps are aligned in a common reference frame.

- **Regression Heads**: Two separate heads output the pointmaps $$X_{1,1}$$ and $$X_{2,1}$$ and associated confidence maps $$C_{1,1}$$ and $$C_{2,1}$$.

- The pointmap calculation involves transforming the outputs from the regression heads into structured 3D pointmaps.

- This includes reformatting and scaling the predictions based on geometric transformations or normalization factors that align the pointmaps in a common coordinate frame or adjust them based on camera intrinsics.

- **Relationship to Camera Geometry**: the pointmap $$X$$ is related to the camera's intrinsic matrix $$K$$ and a depth map $$D$$ through this equation:
  $$
  X_{i,j} = K^{-1} \begin{bmatrix} i \\ j \\ 1 \end{bmatrix} D_{i,j}
  $$
- where, $$X_{i,j}$$ represents the 3D point corresponding to the pixel $$(i, j)$$ in the image.


**Training Objective:**

- **3D Regression Loss**: is designed to minimize the Euclidean distance between predicted and actual 3D point maps. This loss ensures accurate mapping from 2D image pixels to 3D points.

- **Confidence-Aware Loss**: this loss function modifies the regression loss by weighting it with the network's confidence in its predictions. It enhances learning in areas where the network is confident and diminishes the impact of less certain predictions. By prioritizing high-confidence areas, it optimizes the learning process and focuses on regions where the model excels. This method is especially beneficial for handling uncertain or ambiguous image regions, improving training for complex tasks such as 3D reconstruction.


**Downstream Applications:**

- **Point Matching and Camera Intrinsics Recovery**: finds pixel correspondences between images through nearest neighbor searches in 3D pointmap space and optimizes camera intrinsic parameters by aligning 3D points with their image projections.
- **Pose Estimation**: this framework estimates both relative and absolute camera poses using methods like Procrustes alignment or RANSAC with PnP, leveraging the geometric data from pointmaps for 3D mapping.

**Global Alignment:**

- **Framework Integration**: global alignment is a crucial post-processing step that aligns pointmaps from multiple images into a unified 3D space, ensuring consistent and accurate 3D reconstructions across diverse image sets.
- **Process and Optimization**: this method involves constructing a connectivity graph to identify overlapping images, generating pairwise pointmaps, and using an optimization formula to align all pointmaps into a common coordinate frame, with constraints to prevent trivial solutions.
- **Efficiency and its application**: this alignment strategy directly optimizes in 3D space, avoiding traditional reprojection errors and is designed for quick convergence, making it highly suitable for real-time 3D reconstruction applications.

**Depth Estimation Advances, Visual Localization Challenges:**

- This model sets new **state-of-the-art standards** in both **monocular** and **multi-view depth estimation**, outperforming on benchmarks with its innovative **pointmap-based method** for depth prediction from **uncalibrated images**.
- However, this model's performance in **visual localization** is on par with current methods but doesn't surpass top baselines, struggling with the Cambridge Landmarks dataset due to sparse ground-truth pointmaps. The method's absolute camera pose estimation is compromised when camera intrinsics are unknown, resulting in significant errors, as dense ground-truth pointmaps are essential for accurate scaling in real-world localization.

### More experimentation with models in different scenarios:

<iframe src="https://tinyglb.com/viewer/4d16af0f2a6c4514bf537eb4adbff010" style="border: 0; height: 600px; width: 100%"></iframe> 
- 3D Reconstruction from Airbnb Image Sets

<div style="display: flex; justify-content: space-between;">
  <div>
    <iframe src="https://tinyglb.com/viewer/b807183cbd124077874090f637e98dcb" style="border: 0; height: 400px; width: 120%;"></iframe>
    <p style="text-align: center;">Masked Sky (SEAS): Single Photo</p>
  </div>
  <div>
    <iframe src="https://tinyglb.com/viewer/a1741dcdc1cf45b78bfb84df0d4e69dd" style="border: 0; height: 400px; width: 120%;"></iframe>
    <p style="text-align: center;">Unmasked Sky (SEAS)</p>
  </div>
</div>


<iframe src="https://tinyglb.com/viewer/fdfceda838db4dbbb402fdcb42e22340" style="border: 0; height: 600px; width: 100%"></iframe>
This model excels in *impossible matching*: it quickly constructs accurate 3D models from unrelated images, like an unseen office, without requiring camera details or positions.
