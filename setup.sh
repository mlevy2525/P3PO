git submodule update --init --recursive

cd Depth-Anything-V2
git checkout main
mkdir checkpoints
cd checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true
mv depth_anything_v2_metric_hypersim_vitl.pth?download=true depth_anything_v2_metric_hypersim_vitl.pth
cd ../../

cd co-tracker
git checkout main
pip install -e .
pip install matplotlib flow_vis tqdm tensorboard imageio[ffmpeg]
mkdir -p checkpoints
cd checkpoints
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
cd ../../

cd Metaworld
pip install -e .
cd ../

cd dift
pip install xformers
git checkout main
cd ../

cd xarm_env
pip install -e .
cd ../

pip install torchvision
pip install huggingface-hub==0.23.2
pip install --force-reinstall transformers==4.45.2
