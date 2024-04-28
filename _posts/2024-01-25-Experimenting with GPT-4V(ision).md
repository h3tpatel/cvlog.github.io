## Experimenting with GPT-4V(ision)

In this blog, we intend to explore the application of GPT-4 with Vision (GPT-4V) across various mediums, conducting a series of experiments that span medical imaging analysis, LaTeX code conversion from image, comprehension of charts and diagrams, and the identification of object locations.

Let's begin by experimenting with GPT-4V(ision) in Medical Imaging:

**Does GPT4-V amplify the diagnostic precision in medical imaging?**

In this we will use the New England Journal of Medicine (NEJM) Image Challenges as a means to ascertain the accuracy of our model and evaluate its response efficacy.


```text
Prompt: The patient was diagnosed with bacterial endogenous endophthalmitis. Endophthalmitis is a vision-threatening infection of the vitreous and aqueous fluids of the eye. Systemic antimicrobial agents alone will not adequately treat endogenous endophthalmitis. Intravitreal antimicrobial therapy is warranted, and vitrectomy is indicated in cases of severe vision loss or progression of infection despite antimicrobial therapy.

a. Intravitreal antimicrobial injections
b. Intravitreal steroid injection
c. Removal of intraocular lens
d. Topical antimicrobial drops
e. Topicals steroid drops
```

![image-1](https://github.com/h3tpatel/cvlog.github.io/assets/144167031/e1807016-162b-47f1-8bc9-1e1e598b47b0)


```text
GPT4-V: `Correct`

The image you've provided shows a close-up of a person's eyes. The right eye appears normal, with a clear sclera and healthy-looking iris and pupil. However, the left eye exhibits signs of redness and inflammation, particularly in the sclera, which is the white part of the eye. There is also some yellowish discharge in the inner corner of the left eye, which could suggest the presence of an infection or inflammation.

While I cannot diagnose conditions, the symptoms visible in the left eye could be consistent with various eye conditions, including conjunctivitis, episcleritis, or even something more severe like endophthalmitis, which you mentioned in your initial query. Endophthalmitis is a serious condition that requires prompt medical attention. It is characterized by inflammation of the interior of the eye and can be caused by bacterial or fungal infections. The treatment for endophthalmitis typically involves intravitreal injections of antibiotics or antifungals, and in severe cases, a vitrectomy may be necessary.

If this image is related to the initial query about bacterial endogenous endophthalmitis, it is crucial for the individual in the image to receive immediate medical evaluation and treatment from an ophthalmologist. Endophthalmitis can lead to rapid vision loss if not treated promptly and appropriately. The specific treatments you listed, such as intravitreal antimicrobial injections, intravitreal steroid injections, removal of the intraocular lens, topical antimicrobial drops, and topical steroid drops, are all potential components of a comprehensive treatment plan for endophthalmitis, depending on the severity and progression of the infection.
Please note that this assessment is based solely on the visible symptoms in the image and should not be used as a substitute for professional medical advice. If this is a real-life situation, the individual should seek immediate medical attention.
```

As we observe, GPT-4V has been trained on a diverse dataset, leading to a significant reduction in diagnostic errors and an enhancement in the early detection of diseases such as cancer. Accuracy in GPT-4's medical imaging applications is bolstered by its ability to integrate multimodal data, synthesizing information from radiographs, MRIs, and patient medical histories to provide comprehensive diagnostic insights.

Nevertheless, it's essential to rigorously examine its decision-making process to ensure that its selections are based on solid medical logic, rather than mere conjecture.


---

### LaTeX code conversion from image

Another intriguing use case for the GPT-4 with Vision model lies in its ability to convert visual inputs, such as mathematical figures, diagrams, and images, into corresponding code solutions, facilitating a diverse range of problem-solving applications.

```text
Prompt: can you convert this image into latex code
```

<img width="617" alt="image-2" src="https://github.com/h3tpatel/cvlog.github.io/assets/144167031/72a5aa06-0ecf-4e20-bf9c-4be57211ec88">

```text
GPT4-V: 
\documentclass{article}
\usepackage{amsmath}
\begin{document}
\[C = N(d_1)S_t - N(d_2)Ke^{-rt}\]
where
\[d_1 = \frac{\ln\left(\frac{S_t}{K}\right) + \left(r + \frac{\sigma^2}{2}\right)t}{\sigma\sqrt{t}}\]
and
\[d_2 = d_1 - \sigma\sqrt{t}\]
\end{document}
```

<img width="589" alt="image-3" src="https://github.com/h3tpatel/cvlog.github.io/assets/144167031/d8b25c1c-648b-4fd4-a7fa-d8a389ca5864">


```text
Prompt: Draw an elephant using tikz, ensuring to incorporate realistic colors

GPT4-V Complied Version:
```

<img width="950" alt="image-4" src="https://github.com/h3tpatel/cvlog.github.io/assets/144167031/274be9ca-6b5a-4624-923c-f37a2c8838f5">


As we provide GPT-4V, the directive to illustrate an elephant using the TikZ library in LaTeX for visualization, it is indeed capable of generating images based on such straightforward instructions.

<iframe src="https://tinyglb.com/viewer/506ac4b2173a4a75a7c892d7797a0e04" style="border: 0; height: 600px; width: 100%"></iframe>

**Architectures of Unconstrained image-based dense 3D reconstruction:**

How is a point represented in 3D using a pointmap? A pointmap is a dense 2D array of 3D points, expressed as <span style="color:red">$X \in \mathbb{R}^{W \times H \times 3}$</span>, where $W$ and $H$ denote the width and height of the image, respectively.

This allows for a one-to-one mapping between image pixels and 3D scene points, such that each pixel <span style="color:red">$(i, j)$</span> in the image corresponds to a 3D point <span style="color:red">$X_{i,j}$</span> in the scene.

This simplifies the process of extracting geometric data, such as depth and camera parameters, as it allows for a direct mapping from image pixels to their corresponding 3D points, enhancing tasks like multi-view stereo reconstruction without explicit knowledge of camera parameters.

- **Siamese Network with Shared Weights**: Two branches with shared Vision Transformer (ViT) encoders for feature extraction from input image pairs.
- **Transformer Decoders**: Two decoders process features with cross-attention to ensure output pointmaps are aligned in a common reference frame.
- **Regression Heads**: Two separate heads output the pointmaps <span style="color:red">$X_{1,1}$</span> and <span style="color:red">$X_{2,1}$</span> and associated confidence maps <span style="color:red">$C_{1,1}$</span> and <span style="color:red">$C_{2,1}$</span>.
- The pointmap calculation in the architecture entails transforming regression head outputs into 3D pointmaps by reformatting and scaling, incorporating geometric transformations or normalization to standardize coordinates or adjust for camera intrinsics, i.e., using **depthmap <span style="color:red">$D$</span>** and camera intrinsic matrix <span style="color:red">$K^{-1}$</span> to compute 3D points <span style="color:red">$X_{i,j}$</span> for image pixels <span style="color:red">$(i, j)$</span> as <span style="color:red">$X_{i,j} = K^{-1} \begin{bmatrix} i \\ j \\ 1 \end{bmatrix} D_{i,j}$</span>.
