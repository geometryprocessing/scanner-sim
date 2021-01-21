import glob
from pathlib import Path
import os
import igl
import logging
from tqdm.auto import tqdm
import numpy as np
import sys, getopt
import argparse



from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer
from OCCUtils.Topology import Topo, dumpTopology
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Pnt2d
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
from OCC.Core.BRepTools import breptools_UVBounds
from OCC.Core.ShapeFix import ShapeFix_Shape as _ShapeFix_Shape


def load_bodies_from_step_file(pathname, logger=None):
    assert pathname.exists()
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(str(pathname))
    if status == IFSelect_RetDone:  # check status
        shapes = []
        nr = 1
        try:
            while True:
                ok = step_reader.TransferRoot(nr)
                if not ok:
                    break
                _nbs = step_reader.NbShapes()
                shapes.append(step_reader.Shape(nr))  # a compound
                #assert not shape_to_return.IsNull()
                nr += 1
        except:
            logger.error("Step transfer problem: %i"%nr)
            print("No Shape", nr)
    else:
        logger.error("Step reading problem.")
        raise AssertionError("Error: can't read file.")

    return shapes
    
def save_mesh(output_pathname, body, rescale=True, filter_level=0, precision=[0.95, 0.04]):
    mesh = BRepMesh_IncrementalMesh(body, precision[0], False, precision[1], True)
    mesh.SetShape(body)
    mesh.Perform()
    
    top_exp = TopologyExplorer(body)
    brep_tool = BRep_Tool()
    faces = top_exp.faces()
    first_vertex = 0
    tris = []
    verts = []
    #print("Meshing")
    for face in faces:
        face_orientation_wrt_surface_normal = face.Orientation()
        location = TopLoc_Location()
        mesh = brep_tool.Triangulation(face, location)
        if mesh != None:
            # Loop over the triangles
            num_tris = mesh.NbTriangles()
            for i in range(1, num_tris+1):
                index1, index2, index3 = mesh.Triangle(i).Get()
                if not face_orientation_wrt_surface_normal:
                    tris.append([
                        first_vertex + index1 - 1, 
                        first_vertex + index2 - 1, 
                        first_vertex + index3 - 1
                    ])
                else:
                    tris.append([
                    first_vertex + index3 - 1, 
                    first_vertex + index2 - 1, 
                    first_vertex + index1 - 1
                ])
            num_vertices = mesh.NbNodes()
            first_vertex += num_vertices
            for i in range(1, num_vertices+1):
                verts.append(list(mesh.Node(i).Coord()))
    #print(output_pathname)
    if len(verts) > 0:
        verts = np.array(verts)
        tris = np.array(tris)

        if rescale:
            mi, ma = np.min(verts, axis=0), np.max(verts, axis=0)
            mean = (ma - mi)/2 + mi
            verts -= mean#np.mean(verts, axis=0)
            sf = np.max(ma-mi)
            rf = 200#np.random.randint(50, 100)
            verts = verts / sf * rf
        
        if filter_level > 0:
            mi, ma = np.min(verts, axis=0), np.max(verts, axis=0)
            diff = np.sort(ma - mi)
            if filter_level == 1: # Contain if two largest dimensions are roughly similar
                ratio = diff[2]/diff[1]
            if filter_level == 2: # Contain if all dimensions are roughly similar
                ratio = diff[2]/diff[0]
            if ratio <= 2.0 and verts.shape[0] >= 5000:
                igl.write_triangle_mesh(output_pathname, verts, tris)
                
            #print(mean, np.min(verts, axis=0), np.max(verts, axis=0))
        else:
            igl.write_triangle_mesh(output_pathname, verts, tris)
    
def process(data_path="./data/conv/", output_path="./results_fixed", sort=False, filter_level=0, rescale=True, subfolders=False, precision=[0.95, 0.05]):
    
    #print(data_path, output_path, sort, filter_level, rescale, subfolders, precision)
    data_dir = Path(data_path)
    output_dir = Path(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Glob step files in data dir
    extensions = ["stp", "step"]
    step_files = []
    for ext in extensions:
        if subfolders:
            files = [ f for f in data_dir.glob(f"*/*.{ext}")]
        else: 
            files = [ f for f in data_dir.glob(f"*.{ext}")]
        if sort:
            step_files.extend(sorted(files))
        else:
            step_files.extend(files)
        
    #step_files = step_files[rang[0]:rang[1]]
    print(step_files)
    
    # Process step files
    pbar = tqdm(range(len(step_files)))
    for i in pbar:
        pbar.set_description("Processing %s"%step_files[i])
        sf = step_files[i]

        bodies = load_bodies_from_step_file(sf)
        for b_idx, body in enumerate(bodies):
            if os.path.isfile("%s/%s_%04i_mesh.obj"%(output_path, sf.stem, b_idx)):
                continue
            save_mesh("%s/%s_%04i_mesh.obj"%(output_path, sf.stem, b_idx), body, filter_level=filter_level, rescale=rescale, precision=precision)
            
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mesh step files.')
    parser.add_argument('-d', '--data', dest='data_path', type=str, default="data", help='Input path for step files.')
    parser.add_argument('-o', '--output', dest='output_path', type=str, default="meshes", help='Output path for meshes.')
    parser.add_argument('-s', '--sort', dest='sort', type=str2bool, default=True, help='Sort the input files before meshing.')
    parser.add_argument('-r', '--rescale', dest='rescale', type=str2bool, default=True, help='Scale the meshes to the 100.0^3 cube.')
    parser.add_argument('-f', '--filter', dest='filter', type=int, default=0, help='Filter the meshes. 0=No, 1=Largest/Middle<2.0, 2=Largest/Smallest<2.0.')
    parser.add_argument('-l', '--lin_prec', dest='lin_precision', type=float, default=0.95, help='Linear precision of meshing.')
    parser.add_argument('-a', '--ang_prec', dest='ang_precision', type=float, default=0.04, help='Angular precision of meshing.')
    parser.add_argument('-u', '--subfolders', dest='subfolders', type=str2bool, default=False, help='Look for step files in subfolders.')
    a = parser.parse_args()
    
    process(data_path=a.data_path, output_path=a.output_path, sort=a.sort, rescale=a.rescale, 
            filter_level=a.filter, precision=[a.lin_precision, a.ang_precision], subfolders=a.subfolders)