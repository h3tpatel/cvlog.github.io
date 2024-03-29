The ***RAFT (Recurrent All-Pairs Field Transforms)*** and models such as ***CoTracker*** can be used to analyze wildlife, providing insights into the intricacies of animal behavior patterns and nature that can contribute significantly to our understanding and conservation efforts:

1. **Animal Behavior Analysis**: Both RAFT and CoTracker can be used to analyze wildlife to study their animal behavior patterns. By accurately tracking movement over time, researchers can study and gain insights into migration patterns, feeding behavior, mating rituals, and social interactions, which will be crucial for effective conservation strategies.
2. **Habitat, Climate, and Ecological Insights**: By leveraging this model, changes in wildlife habitats can be monitored more effectively. Analyzing a series of satellite images allows for the detection of alterations in water levels, human encroachment, and vegetation changes. Having these crucial data points allows us to decode the intricate relationship between climate change and its myriad impacts.
3. **Research on Animal Locomotion**: Understanding how animals move within certain habitats can assist in the creation of environments that support their natural behavior and facilitate the movement of wildlife. This research is essential for designing reserves and conservation areas that align with the needs of various species.
4. **Anti-Poaching Surveillance**: Employing surveillance systems designed to track multiple points can help detect unusual activities that may indicate poaching. By distinguishing between animal movements and human intrusion, conservationists can respond more rapidly to protect endangered species.


<p float="left">
  <img src="https://github.com/h3tpatel/cvlog.github.io/assets/144167031/71486c56-fe33-4bf2-9d55-c4af4073c441" width="49%" />
  <img src="https://github.com/h3tpatel/cvlog.github.io/assets/144167031/839f4cb5-5d6c-4c4b-b2b2-98011d01303b" width="49%" />
</p>
<p float="left">
  <img src="https://github.com/h3tpatel/cvlog.github.io/assets/144167031/580d979f-46ae-4f2a-82e9-e8d161796fd0" width="49%" />
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

**Regular grid + Segmentation mask**

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

