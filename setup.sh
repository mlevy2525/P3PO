cd Depth-Anything-V2
mkdir checkpoints
cd checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true
mv depth_anything_v2_metric_hypersim_vitl.pth?download=true depth_anything_v2_metric_hypersim_vitl.pth
cd ../../

cd co-tracker
pip install -e .
pip install matplotlib flow_vis tqdm tensorboard imageio[ffmpeg]
mkdir -p checkpoints
cd checkpoints
wget https://huggingface.co/facebook/cotracker/resolve/main/cotracker2v1.pth
cd ../../