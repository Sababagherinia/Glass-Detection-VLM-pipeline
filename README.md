# Glass-Detection-VLM-pipeline
# RGB-D -> VLM -> Mapping pipeline
This repository contains a minimal skeleton pipeline to read RGB-D frames, process them with a visual-language model (e.g CLIP via Hugging Face `transformers`), and accumulate a 3D map (Open3D output, finally pyoctomap integration).

there is a folder `data` with subfolders `rgb` and `depth`. Place matching RGB images in `data/rgb` and depth images in `data/depth` (16-bit PNG in millimeters or float meters accepted).

Notes
- The code uses `transformers` CLIP weights and `open3d` for point cloud handling.
- `pyoctomap` integration is optional — the code contains a placeholder where you can integrate OctoMap insertion calls.
- For RealSense live capture, install `pyrealsense2` and run with `--mode realsense`.

Next steps
- Add region proposals / object detection and crop detected regions before passing to CLIP to get object-level labels.
- Integrate `pyoctomap` properly: insert point clouds using known sensor origin transforms and use occupancy queries to build a traversability map.
- Add a small unit test that loads one RGB-D pair and ensures the pipeline runs end-to-end.