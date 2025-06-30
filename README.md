# Automated tooth crowding analysis
Deep learning to determine Little Irregularity Index on occlusal intra-oral photographs

## Method

![Method](docs/method.png)

The method was developed with keypoint annotations on intra-oral scans and corresponding intra-oral photographs. The anatomic contact points were registered from the scan to the photo and YOLOv8 was used to determine tooth objects. Each tooth was predicted with a segmentation of its boundary, the mesial and distal anatomic contact points, and the physical size of the tooth in millimeters.

## Installation

Install a Conda environment:

``` bash
conda create -n crowding python=3.11
conda activate crowding
```

Install the Pip requirements:

``` bash
pip install -r requirements.txt
```

Install Ultralytics YOLOv8

``` bash
pip install -e ultralytics
```

## Inference

You can run inference on the photograph in the figure above by running `infer.py`. Measurements of the anterior teeth will be saved to '*measurements.xlsx*'. The model predictions can be visualized by running `infer.py --verbose`. Furthermore, the model can be run on your own photographs by specifying a folder with images using the `in_dir` argument.


## Citation

``` bib
@article{crowding_ai,
    author = {Hertig, Gabriel and van Nistelrooij, Niels and Schols, Jan and Xi, Tong and Vinayahalingam, Shankeeth and Patcas, Raphael},
    title = {Quantitative tooth crowding analysis in occlusal intra-oral photographs using a convolutional neural network},
    journal = {European Journal of Orthodontics},
    volume = {47},
    number = {3},
    year = {2025},
    month = {05},
    doi = {10.1093/ejo/cjaf025},
}
```
