# Autotuning of Screened Poisson Surface Reconstruction Parameters

## Requirements

C++:
- Poisson surface reconstruction executable compiled from [source](https://github.com/mkazhdan/PoissonRecon) (commit 28934e8798)

Python:
- plotoptix 0.13.2
- matplotlib 3.3.4
- numpy 1.19.4
- pymeshlab 0.1.8
- pyntcloud 0.1.3
- trimesh 3.9.1

If you are working with pip or conda, you can easily set up a proper environment with the Python dependencies.

### pip
```
pip install -r requirements.txt
```

### conda
```
conda env create -f environment.yml
conda activate sls-poisson
```

## Autotuning

The `sweep.py` script performs the autotuning procedure and a call might look like this

```
python sweep.py --executable path/to/poisson_executable --input path/to/input_pointcloud.ply --output_dir out/object_name --reference path/to/reference_mesh.obj --parameter depth 6 10 5 --parameter samplesPerNode 1 20 5
```
`path/to/poisson_executable` is the Poisson reconstruction executable compiled as part of the requirements, `path/to/input_pointcloud.ply` is the point cloud (with normals) from the SLS reconstruction pipeline and `path/to/reference_mesh.obj` is the reference/target mesh used for evaluating the reconstruction quality. The parameters used for sweeping are provided as arguments like `--parameter name min_value max_value step`, where `name` is the name of the parameter in the Poisson reconstruction executable.

The calls used for the sweep in the paper are

Pawn:
```
python sweep.py --executable path/to/poisson_executable --input ./data/pawn_group.ply --trim_y -76 70 --parameter depth 5 11 7 --parameter samplesPerNode 1 20 5 --output_dir ./out/pawn --reference ./data/pawn_reference_mesh.obj
```

Rook:
```
python sweep.py --executable path/to/poisson_executable --input ./data/rook_group.ply --trim_y -110 67 --parameter depth 5 11 7 --parameter samplesPerNode 1 20 5 --output_dir ./out/rook --reference ./data/rook_reference_mesh.obj
```

Note that the evaluation is not deterministic (in terms of exact numbers) as the samples used for computing the Hausdorff distance are randomly selected from the surface.

## Applying the Parameters

The discovered parameters can be directly used with the PSR executable to reconstruct the surface of a test object.

The surface of the avocado object from the TPOD was reconstructed with the following command
```
path/to/poisson_executable --in ./data/avocado_group.ply --depth 9 --samplesPerNode 20 --out ./out/avocado_mesh.ply 
```