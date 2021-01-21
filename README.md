# scanner-sim

structure:
- scanner
  - calibration (Setup-Intrinsic - JSON (add apertures) + Predistorted/Premodulated Pattern images)
- data
  - calibrations: for simulation and scanner
  - configs: configuration files for rendering
  - datasets: processed datasets for data driven applications
  - models: CAD reference models
  - objects: meshed CAD and other models
  - patterns: coding patterns for sls
  - results: rendering results
  - scenes: virtual scenes for rendering
- simulator (depends on Setup-Intrinsic parameters from real-world calibration and Object-Extrinsic parameters for given scan: shape/position/material of the object; can output groundtruth)
- processing: 
  - validation(just measures the metrics based on real-world and simulated images)
  - refinement (Object-Extrinsic with simple gradient descent: position for sure, maybe camera/projector calibration - danger of overfitting)
- reconstruction (takes decoded images & calibration, outputs point cloud & depthmap)
  - estimation (Object-Extrinsic if ground-truth geometry is known)
- decoding takes images and generates horizontal and vertical correspondence map
- applications
  - (AI) denoising
  - (AI) hole_filling
  - (Future) shape/material optimization (regular gradient descent or with differentiable renderer if too many parameters)
  
  
  
steps:
- meshing: python3 processing/meshing.py -d data/models/abc_00 -o data/objects/abc_filter1 -s True -r True -f 1 -a 0.04 -u True -b 0 -e 2000
- rendering: 



TODO:
1. Batchrender folder: one job per object
1.1 Render object, multiple rotations per object, one pattern
2. Cleanup scenes <include filename="nested-scene.xml"/>
3. Cleanup config
