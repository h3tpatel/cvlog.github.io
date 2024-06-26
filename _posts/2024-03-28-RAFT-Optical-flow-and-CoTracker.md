The ***RAFT (Recurrent All-Pairs Field Transforms)*** and models such as ***CoTracker*** can be used to analyze wildlife, providing insights into the intricacies of animal behavior patterns and nature that can contribute significantly to our understanding and conservation efforts:

1. **Animal Behavior Analysis**: Both RAFT and CoTracker can be used to analyze wildlife to study their animal behavior patterns. By accurately tracking movement over time, researchers can study and gain insights into migration patterns, feeding behavior, mating rituals, and social interactions, which will be crucial for effective conservation strategies.
2. **Habitat, Climate, and Ecological Insights**: By leveraging this model, changes in wildlife habitats can be monitored more effectively. Analyzing a series of satellite images allows for the detection of alterations in water levels, human encroachment, and vegetation changes. Having these crucial data points allows us to decode the intricate relationship between climate change and its myriad impacts.
3. **Research on Animal Locomotion**: Understanding how animals move within certain habitats can assist in the creation of environments that support their natural behavior and facilitate the movement of wildlife. This research is essential for designing reserves and conservation areas that align with the needs of various species.
4. **Anti-Poaching Surveillance**: Employing surveillance systems designed to track multiple points can help detect unusual activities that may indicate poaching. By distinguishing between animal movements and human intrusion, conservationists can respond more rapidly to protect endangered species.

<p float="left">
  <img src="https://github.com/h3tpatel/cvlog.github.io/assets/144167031/22333368-b02a-4d2f-b3f2-e03a9ace552f" width="49%" />
  <img src="https://github.com/h3tpatel/cvlog.github.io/assets/144167031/71486c56-fe33-4bf2-9d55-c4af4073c441" width="49%" />
</p>
<p float="left">
  <img src="https://github.com/h3tpatel/cvlog.github.io/assets/144167031/b3b90dd9-668b-4dc4-8b70-4655ab3fcaab" width="49%" />
  <img src="https://github.com/h3tpatel/cvlog.github.io/assets/144167031/964cdf23-ed69-4ace-9154-0e3f56de9f4d" width="49%" />
</p>

Now, let's begin by implementing the RAFT model to observe such use cases. First, we load the RAFT model and set it to evaluation mode, initializing it with the pre-trained weights **`(Raft_Large_Weights.C_T_SKHT_V2)`**.

```python
model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(device)
model = model.eval()
```

After that, we preprocess each pair of consecutive frames, applying transformations to normalize and resize them, preparing them for optical flow estimation. Then, the frames are permuted to change their shape from **[N, H, W, C] to [N, C, H, W]**.

```python
def preprocess(batch):
    transforms = T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
        T.Resize(size=(520, 960)),
    ])
    return transforms(batch)

# preprocess video frames
frames, _, _ = read_video(str(video_path), pts_unit='sec')
frames = frames.permute(0, 3, 1, 2)  # *(N, H, W, C) -> (N, C, H, W)*
```

Finally, the code iterates over pairs of consecutive frames, saving the estimated optical flow from **`model(img1, img2)`**, which is converted to an RGB image using **`flow_to_image`** to visualize the flow vectors. These images are then converted to PIL images.

```python
for i, (img1, img2) in enumerate(zip(frames, frames[1:])):
    img1 = preprocess(img1[None]).to(device)
    img2 = preprocess(img2[None]).to(device)

    list_of_flows = model(img1, img2)
    predicted_flow = list_of_flows[-1][0]
    flow_img = flow_to_image(predicted_flow).to("cpu")

    # convert tensor to a PIL Image
    pil_img = F.to_pil_image(flow_img)
    pil_img.save(output_folder / f"predicted_flow_{i}.jpg")
```
-------
#### **New Insights into Animals: Pollination to Migration, Flight Dynamics to Social Behavior**

#### **Insights into Hummingbird**
<p float="left">
  <img src="https://github.com/h3tpatel/cvlog.github.io/assets/144167031/bda57d14-5ac9-4caa-bf93-65a125087c49" width="49%" />
  <img src="https://github.com/h3tpatel/cvlog.github.io/assets/144167031/bd04a005-80b1-4b0a-b25a-6071dff92abb" width="49%" />
</p>

1. **Hummingbird movement tracking**: As shown in the trained model on hummingbird videos, we can track the movement of hummingbirds as they fly between flowers.
2. **Hummingbird-plant relationships**: Analyzing these videos yields insights into the *fascinating mutualistic relationships hummingbirds have with the plants they pollinate*. Studying hummingbird and flower morphologies can help explain the interaction patterns seen in hummingbird-plant networks.
3. **Flight dynamics**: Another use case is quantifying hummingbird *flight dynamics*, including their complex wing kinematics during hovering and maneuvering. 
4. **Individual identification and population monitoring**: By identifying individuals based on their *unique plumage patterns* and tracking them across camera feeds, we could monitor populations, reveal movement patterns, and estimate abundances.


#### **Insights into Humpback Whales**
<p float="left">
  <img src="https://github.com/h3tpatel/cvlog.github.io/assets/144167031/3a907355-def9-4e1e-a1cc-d2498c749672" width="49%" />
  <img src="https://github.com/h3tpatel/cvlog.github.io/assets/144167031/e4841d4f-eb58-4aae-bab9-7cc0f89ba642" width="49%" />
</p>


1. **Migration and behavior analysis**: Track migration routes, timing, and corridors using fluke photos, and classify surface behaviors like breaching and slapping to study communication, social interactions, cultural transmission, and responses to human activities and climate change.
2. **Disturbance detection and conservation**: Detect changes in behavior, movement, or habitat usage that may indicate disturbances from human activities, helping monitor threats and inform conservation efforts.
3. **Comprehensive modeling at scale**: Integrate acoustic, visual, environmental, and movement data to build comprehensive behavioral models and enable large-scale analysis, revealing population-level and long-term trends.

<p float="left">
  <img src="https://github.com/h3tpatel/cvlog.github.io/assets/144167031/5f620f98-cd75-4131-8d51-7296e6852378" width="49%" />
  <img src="https://github.com/h3tpatel/cvlog.github.io/assets/144167031/bb9bb598-2325-4d36-9dd5-714969a0f7a8" width="49%" />
</p>

#### **CoTracker Implementation: Regular grid + Visualize Track Traces**

Now, let's implement the CoTracker model, which is designed for tracking points (pixels) across video frames. First, we specify the path of the video and convert it into a PyTorch tensor for further preprocessing. The video tensor is permuted to match the expected input **`(Batch x Time x Channels x Height x Width)`**. Then, we initialize the model using pre-trained checkpoints.

```python
local_video_path = '/content/.mp4'
video = read_video_from_path(local_video_path)
video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
```

After that, we run inference on the video with a specified **`grid_size`** parameter, which determines the density of the tracking grid over several video frames. It then returns the predicted tracks **`pred_tracks`** and their visibility **`pred_visibility`** across frames.

```python
grid_size = 50
pred_tracks, pred_visibility = model(video, grid_size=grid_size)
```

Finally, we utilize the **`Visualizer`** class to generate a visual representation of the tracking.

```python
vis = Visualizer(save_dir='/content/', pad_value=100)
vis.visualize(video=video, tracks=pred_tracks, visibility=pred_visibility, filename='video')
```
