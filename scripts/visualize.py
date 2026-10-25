import os
import sys
from pathlib import Path


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "raft", "core"))
)  # noqa: PTH100, PTH118, PTH120
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voxelmorph_folder"
    )
)  # noqa: PTH100, PTH118, PTH120
from flow_visualizer import FlowVisualizer
from my_utils.configs import HEIGHT, TEST_DATA, WIDTH
from my_utils.image_utils import interpole_and_pad
import tifffile as tiff
import numpy as np


video_path = sorted(Path(TEST_DATA).iterdir())
flow_map_path = Path("outputs/rasor-cnn/output_20260707_103416/results_gap1/flowmaps")

# Load flow maps
flow_maps = {}
for flow in flow_map_path.glob("*.npy"):
    name = flow.stem.strip("()")
    key = (int(name.split(",")[0]), int(name.split(",")[1]))
    flow_maps[key] = np.load(flow)

video = []
# Load video
for idx in range(min((len(video_path), 100))):
    result = {}
    # Load moving and fixed images
    # Read images
    img = tiff.imread(video_path[idx])
    # Interpole/Pad images to HEIGHT, WIDTH if needed
    img = interpole_and_pad(img, [HEIGHT, WIDTH])
    video.append(np.expand_dims(img, axis=-1))

visualizer = FlowVisualizer(video, flow_maps, grid_step=20, gap=1)
visualizer.run()
