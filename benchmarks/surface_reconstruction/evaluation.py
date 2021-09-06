from pathlib import Path
import pymeshlab as ml

from typing import Optional, Union

def evaluate(reference_mesh_path: Union[Path, str], reconstruction_mesh_path: Union[Path, str], num_samples: int=1000000, output_path: Optional[Union[Path, str]] = None, min_value: float = 0, max_value: float = 2):
    """ Evaluate how close a mesh reconstructed from a scan is to a reference mesh

    Args:
        reference_mesh_path: Path to the reference mesh
        reconstruction_mesh_path: Path to the mesh reconstructed from a scan
        num_samples: Number of samples on the surface used for evaluation
        output_path (optional): Path where the evaluation samples will be saved. They will be color mapped by the Hausdorff distance.
        min_value: Minimum value used when color mapping the Hausdorff distance.
        max_value: Maximum value used when color mapping the Hausdorff distance.
    
    Returns:
        Dictionary with the following metrics
        - RMS: Root mean squared error from the reconstruction to the reference
        - min: Minimum distance from the reconstruction to the reference
        - max: Maximum distance from the reconstruction to the reference
        - mean: Mean distance from the reconstruction to the reference
        - hausdorff: One-sided Hausdorff distance from the reconstruction to the reference
    """

    reference_mesh_path = Path(reference_mesh_path).absolute()
    reconstruction_mesh_path = Path(reconstruction_mesh_path).absolute()

    mesh_set = ml.MeshSet()

    mesh_set.load_new_mesh(str(reference_mesh_path))
    reference_mesh_id = mesh_set.current_mesh_id()

    mesh_set.load_new_mesh(str(reconstruction_mesh_path))
    reconstruction_mesh_id = mesh_set.current_mesh_id()

    statistics = mesh_set.apply_filter("hausdorff_distance", sampledmesh=reconstruction_mesh_id, targetmesh=reference_mesh_id, samplevert=False, sampleface=True, samplenum=num_samples, savesample=True)
    eval_samples_id = 3
    mesh_set.set_current_mesh(eval_samples_id)

    if output_path is not None:
        output_path = Path(output_path).absolute()
        # Colormap the sample points (mesh index 3) with the Hausdorff distance
        mesh_set.apply_filter("quality_mapper_applier", minqualityval=min_value, maxqualityval=max_value)
        mesh_set.save_current_mesh(str(output_path))

    statistics['hausdorff'] = statistics['max']

    return statistics

# evaluate("./data/rook/reference_mesh.obj", "./out/rook/rook_group_poisson_depth-5_samplesPerNode-1.ply", output_path="samples.ply")