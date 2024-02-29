# Blur and Beyond: Implementing and Analyzing Adversarial Attacks on Images with Gaussian Blur, FGSM, and PGD

## Basic Implementation of Model Loading and Image Operation

1. Let’s start by implementing standard libraries and defining the device settings

```python
!pip install torch torchvision


import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

2. Preparing the Model: We will use ResNet-50 CNN dataset (50-layer convolutional neural network: 48 convolutional layers, one MaxPool layer, and one average pool layer) pre-trained model to perform adversarial attacks.

```python
model = models.resnet50(pretrained=True)
model.eval()
```

3. Now, let’s start by defining the class index mapping: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a

```python
# Define the idx_to_class mapping based on the ImageNet classes
idx_to_class = {
    0: 'tench',
    1: 'goldfish',
    # ... (other class mappings) ...
    276: 'hyena, hyaena',
    388: 'giant panda',
    # ... (other class mappings) ...
    669: 'mosquito net',
    # ... (other class mappings) ...
}
```

4. After loading the original image and converting it to RGB, the `preprocess` variable applies a series of transformations: resizing the image to $224 \times 224$ pixels and converting it to a PyTorch tensor. The `unsqueeze(0)` function is then used to add a batch dimension, preparing the image tensor for model input.

```python
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

original_image = Image.open('/content/panda.jpeg').convert('RGB')
image_tensor = preprocess(original_image).unsqueeze(0).to(device)

original_output = model(image_tensor)
_, original_pred = torch.max(original_output, 1)
original_class = 'Panda'
label_index = 388
```

## Types of Adversarial Attacks

### Gaussian blur

Now, we start by implementing basic adversarial attack like Gaussian Blur. First, we start by defining `transforms.GaussianBlur` function, which applies a Gaussian blur using a $5 \times 5$ kernel and a standard deviation range of $0.1$ to $2.0$.This process blurs the image by averaging pixel values using a Gaussian distribution.

```python
gaussian_blur = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
```
After, applying the Gaussian blur to `image_tensor` smooths the image, reducing detail and noise by using the Gaussian kernel. And we add the noise to the original image tensor from the blurred image tensor.

```python
blurred_image_tensor = gaussian_blur(image_tensor)
noise = blurred_image_tensor - image_tensor
```

This blurred image tensor is then fed into the model to predict the class, with `torch.max` identifying the most likely class from the output. After, `idx_to_class[blurred_pred.item()]` translates the predicted class index into a human-readable class label using a predefined `idx_to_class` dictionary that associates indices with class names.

```python
blurred_output = model(blurred_image_tensor)
_, blurred_pred = torch.max(blurred_output, 1)
blurred_class = idx_to_class[blurred_pred.item()]
```

The code uses PIL and matplotlib to create a $1\times 3$ subplot figure, convert PyTorch tensors to PIL images for display, and print the classification results of the processed images.

```python
# Visualize the original, blurred, and noise images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(transforms.ToPILImage()(image_tensor.squeeze().cpu()))
plt.title(f'Original Image: {original_class}')

plt.subplot(1, 3, 2)
plt.imshow(transforms.ToPILImage()(blurred_image_tensor.squeeze().cpu()))
plt.title(f'Blurred Image: {blurred_class}')

plt.subplot(1, 3, 3)
plt.imshow(transforms.ToPILImage()(noise.squeeze().cpu()))
plt.title('Noise')

plt.show()

# print out the classification results
print(f'Original Image classified as: {original_class}')
print(f'Blurred Image classified as: {blurred_class}')
```

#### Output:

![gaussian-blur](https://github.com/h3tpatel/cvlog.github.io/assets/144167031/39bc3630-7c8b-4f0f-840e-4a56d7e5ba94)

### Fast Gradient Sign Attack (FGSM)

Now, let's implement the Fast Gradient Sign Method (FGSM), a type of adversarial attack on a neural network model. First, we begin by switching the model to evaluation mode and enabling gradient calculations by setting `requires_grad` to `True` on the image tensor. Next, we compute the model's output for the image and the loss relative to the actual label, followed by backpropagation to calculate the gradients with respect to the loss.

```python
model.eval()
image_tensor.requires_grad = True
output = model(image_tensor)

# calculate the loss
loss = nn.CrossEntropyLoss()(output, torch.tensor([label_index]).to(device))

# zero all existing gradients
model.zero_grad()

# calculate gradients of model in backward pass
loss.backward()

# collect datagrad
data_grad = image_tensor.grad.data
```

After that, we call the FGSM attack, which perturbs the original image along the data gradient's sign, then feeds it to the model to obtain the `perturbed_class` from the `idx_to_class` dictionary using the new prediction index.

```python
# call FGSM Attack
epsilon = 0.1
sign_data_grad = data_grad.sign()
perturbed_image = image_tensor + epsilon * sign_data_grad

output = model(perturbed_image)
_, perturbed_pred = torch.max(output, 1)

# convert the index to the corresponding class label
perturbed_class = idx_to_class[perturbed_pred.item()]
```

Now, let's visualize the original and perturbed images

```python
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(transforms.ToPILImage()(image_tensor.squeeze().cpu()))
plt.title(f'Original Image: {original_class}')

plt.subplot(1, 2, 2)
plt.imshow(transforms.ToPILImage()(perturbed_image.squeeze().cpu()))
plt.title(f'Perturbed Image: {perturbed_class}')

plt.show()

print(f'Original Image classified as: {original_class}')
print(f'Perturbed Image classified as: {perturbed_class}')
```

#### Output:

![fgsm](https://github.com/h3tpatel/cvlog.github.io/assets/144167031/ad0d6dfd-c6a2-44f8-9cff-3b502fde5b96)

### Projected Gradient Descent Attack (PGD)

Let's begin by implementing a sophisticated adversarial technique. We initialize a class with a neural network model and specific attack parameters. The class's `perturb` method then generates adversarial examples by iteratively modifying the input images to maximize the classification loss, while ensuring the perturbations remain within a defined epsilon constraint. 

```python
# Define the PGD Attack Class
class PGDAttack:
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, random_start=True):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def perturb(self, images, labels):
        images = images.clone().detach().to(device)
        labels = labels.clone().detach().to(device)
        if self.random_start:
            images = images + torch.empty_like(images).uniform_(-self.eps, self.eps)
            images = torch.clamp(images, 0, 1)

        loss = nn.CrossEntropyLoss()

        for _ in range(self.steps):
            images.requires_grad = True
            outputs = self.model(images)
            model.zero_grad()
            cost = loss(outputs, labels).to(device)
            cost.backward()

            adv_images = images + self.alpha*images.grad.sign()
            eta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            images = torch.clamp(images + eta, min=0, max=1).detach_()

        return images
```

After that, we intiate `PGDAttack` and use it to generate an adversarial image from the original image tensor and label index. And then compute the noise added by subtracting the original image from the adversarial image

```python
pgd_attack = PGDAttack(model)
adv_image = pgd_attack.perturb(image_tensor, torch.tensor([label_index]).to(device))

noise = adv_image - image_tensor
```

After, the adversarial image is passed through the model to get the predicted class, which is then mapped to a human-readable label using `idx_to_class`.

```python
adv_output = model(adv_image)
_, adv_pred = torch.max(adv_output, 1)
adv_class = idx_to_class[adv_pred.item()]
```

Now, let's visualize the original, adversarial, and noise images

```python
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(transforms.ToPILImage()(image_tensor.squeeze().cpu()))
plt.title(f'Original Image: {original_class}')

plt.subplot(1, 3, 2)
plt.imshow(transforms.ToPILImage()(adv_image.squeeze().cpu()))
plt.title(f'Adversarial Image: {adv_class}')

plt.subplot(1, 3, 3)
plt.imshow(transforms.ToPILImage()(noise.squeeze().cpu()))
plt.title('Noise')

plt.show()

# Print out the classification results
print(f'Original Image classified as: {original_class}')
print(f'Adversarial Image classified as: {adv_class}')
```
#### Output: 

![pgd](https://github.com/h3tpatel/cvlog.github.io/assets/144167031/c3127ef1-bdf6-45ed-8ea4-8e1c4cedbc11)
