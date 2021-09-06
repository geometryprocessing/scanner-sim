from argparse import ArgumentParser
import itertools
import numpy as np
import pymeshlab as ml
from multiprocessing import Pool, cpu_count
from pathlib import Path
import subprocess

def execute(cmd, mesh_path, trim_y, index, max_num):
    print("Reconstructing {} ({}/{})...".format(mesh_path.name, index + 1, max_num))
    result = subprocess.check_call(cmd)
    if result != 0:
        print("Reconstruction {} failed ({}/{})...".format(mesh_path.name, index + 1, max_num))

    # Trim the mesh below and above a certain height
    if trim_y:
        mesh_set = ml.MeshSet()
        mesh_set.load_new_mesh(str(mesh_path))
        mesh_set.apply_filter("conditional_face_selection", condselect="((y0 < {y_min:}) || (y0 > {y_max:})) || ((y1 < {y_min:}) || (y1 > {y_max:})) || ((y2 < {y_min:}) || (y2 > {y_max:}))".format(y_min=trim_y[0], y_max=trim_y[1]))
        mesh_set.apply_filter("delete_selected_faces")
        mesh_set.apply_filter("conditional_vertex_selection", condselect="(y < {y_min:}) || (y > {y_max:})".format(y_min=trim_y[0], y_max=trim_y[1]))
        mesh_set.apply_filter("delete_selected_vertices")
        mesh_set.save_current_mesh(str(mesh_path))

def evaluate(eval_dir, mesh_path, reference_mesh_path, parameter_values, index, max_num):
    mesh_set = ml.MeshSet()

    mesh_set.load_new_mesh(str(mesh_path))
    reconstruction_mesh_id = mesh_set.current_mesh_id()

    mesh_set.load_new_mesh(str(reference_mesh_path))
    reference_mesh_id = mesh_set.current_mesh_id()

    eval_samples_path = eval_dir / "{}_samples.ply".format(Path(mesh_path).stem)
    eval_samples_id = 3
    if not eval_samples_path.exists():
        stats_xy = mesh_set.apply_filter("hausdorff_distance", sampledmesh=reconstruction_mesh_id, targetmesh=reference_mesh_id, samplevert=False, sampleface=True, samplenum=1000000, savesample=True)
        mesh_set.set_current_mesh(eval_samples_id)
    else:
        mesh_set.load_new_mesh(str(eval_samples_path))
        stats_xy = mesh_set.apply_filter("per_vertex_quality_stat")

    # Colormap the sample points (mesh index 3), transfer the mesh normals and save them
    mesh_set.apply_filter("quality_mapper_applier", minqualityval=0, maxqualityval=2) #maxqualityval=4
    #mesh_set.apply_filter("vertex_attribute_transfer", sourcemesh=reconstruction_mesh_id, targetmesh=eval_samples_id, colortransfer=False, normaltransfer=True)
    mesh_set.save_current_mesh(str(eval_samples_path))
    
    # # Parametrize the reconstruction mesh.
    # # This operation might fail for some meshes but that's ok (we don't want to show all of them)
    # try:
    #     mesh_set.set_current_mesh(reconstruction_mesh_id)
    #     mesh_set.apply_filter("parametrization_trivial_per_triangle", textdim=4096, border=8)

    #     # Generate a texture from the distance samples and copy it to the right directory
    #     eval_texture_path = eval_dir / "{}.obj.png".format(Path(mesh_path).stem)
    #     mesh_set.apply_filter("transfer_vertex_attributes_to_texture_1_or_2_meshes", sourcemesh=eval_samples_id, targetmesh=reconstruction_mesh_id, attributeenum="Vertex Color", 
    #                         textw=4096, texth=4096, textname=eval_texture_path.name)
    #     (mesh_path.parent / eval_texture_path.name).replace(eval_texture_path)

    #     # FIXME: Colors in texture are linear RGB

    #     # Save the evaluation mesh
    #     eval_mesh_path = eval_dir / "{}.obj".format(Path(mesh_path).stem)
    #     mesh_set.save_current_mesh(str(eval_mesh_path))

    #     # HACK: Patch the OBJ to use a relative material lib path
    #     obj_lines = []
    #     with open(eval_mesh_path) as f:
    #         obj_lines = f.readlines()
        
    #     mtllib_line_index = [i for i, s in enumerate(obj_lines) if 'mtllib' in s][0]
    #     obj_lines[mtllib_line_index] = 'mtllib {}.mtl\n'.format(eval_mesh_path.name)
    #     obj_lines.insert(mtllib_line_index + 1, "usemtl material_0\n")

    #     with open(eval_mesh_path, "w") as f:
    #         f.writelines(obj_lines)
    # except:
    #     print("Parametrization failed: {}".format(mesh_path.name))

    # _ = mesh_set.apply_filter("distance_from_reference_mesh", measuremesh=reconstruction_mesh_id, refmesh=reference_mesh_id, signeddist=False)
    # mesh_set.set_current_mesh(reconstruction_mesh_id)
    # mesh_set.apply_filter("quality_mapper_applier", minqualityval=0, maxqualityval=4)
    # mesh_set.save_current_mesh(str(eval_dir / mesh_path.name))

    print("Evaluated mesh {} ({}/{})".format(mesh_path.name, index + 1, max_num))

    return { 'hausdorff': stats_xy['max'] }

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True, help="Path to the PoissonRecon executable")
    parser.add_argument("--input", type=Path, required=True, help="Input point cloud")
    parser.add_argument("--reference", type=Path, help="Path to the reference mesh used for evaluation")
    parser.add_argument("--output_dir", type=Path, default="out", help="Output directory")
    parser.add_argument('--parameter', default=[], action='append', nargs=4, required=True, help="Parameter to perform sweep on (format: name min max steps)")
    parser.add_argument("--trim_y", type=int, default=None, nargs=2, help="Min/max y")
    args = parser.parse_args()

    # Extract names of the parameters and the ranges to sweep
    parameter_names = []
    parameter_ranges = []
    for name, min_value, max_value, num_steps in args.parameter:
        parameter_names.append(name)

        min_value = float(min_value) if ('.' in min_value) else int(min_value)
        max_value = float(max_value) if ('.' in max_value) else int(max_value)
        num_steps = int(num_steps)

        value_type = type(min_value)
        parameter_ranges.append(np.linspace(min_value, max_value, num_steps).astype(value_type))

    # Get all combinations of parameter values
    combinations = list(itertools.product(*parameter_ranges))
    num_combinations = len(combinations)

    # Reconstruct surfaces
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh_paths = []
    with Pool(processes=cpu_count()) as pool:
        for combination_index, values in enumerate(combinations):
            # Build the result file path
            mesh_filename = "{}_poisson".format(args.input.stem)
            for value_index, value in enumerate(values):
                parameter_name = parameter_names[value_index]
                mesh_filename += "_{}-{}".format(parameter_name, str(value))
            mesh_filename += ".ply"
            mesh_path = args.output_dir / mesh_filename
            mesh_paths += [ mesh_path ]

            if mesh_path.exists():
                print("Skipped reconstruction {} ({}/{})...".format(mesh_filename, combination_index + 1, num_combinations))
                continue

            # Build the command
            cmd = [str(args.executable), "--in", str(args.input)]
            for value_index, value in enumerate(values):
                parameter_name = parameter_names[value_index]
                cmd += ["--{}".format(parameter_name), str(value)]
            cmd += ["--out", str(mesh_path)]

            pool.apply_async(execute, args=[cmd, mesh_path, args.trim_y, combination_index, num_combinations])

        pool.close()
        pool.join()

    if args.reference:
        eval_dir = args.output_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        with Pool(processes=cpu_count()) as pool:
            stats = pool.starmap(evaluate, [[eval_dir, mesh_path, args.reference, combinations[i], i, num_combinations] for i, mesh_path in enumerate(mesh_paths)])

        # Print out the parameters and their ranges
        for name, values in zip(parameter_names, parameter_ranges):
            print(name, values)

        # Collect and print out the metrics for all combinations
        metrics = {k: [] for k in stats[0].keys()}
        for stat in stats:
            for k, v in stat.items():
                metrics[k] += [v]
        
        for k, v in metrics.items():
            print(f"{k}:", *["{:.3f}".format(round(value, 3)) for value in v])

        for k, v in metrics.items():
            print(f"{k}:", " & ".join(["{:.3f}".format(round(value, 3)) for value in v]))


