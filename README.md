# scanner-sim

Folders Structure:
- Scanner
  - Calibration (Setup-Intrinsic - JSON (add apertures) + Predistorted/Premodulated Pattern images)
- Data
- Simulator (depends on Setup-Intrinsic parameters from real-world calibration and Object-Extrinsic parameters for given scan: shape/position/material of the object; can output groundtruth)
- (Optional) Validation (just measures the metrics based on real-world and simulated images)
  - Refinement (Object-Extrinsic with simple gradient descent: position for sure, maybe camera/projector calibration - danger of overfitting)
- Reconstruction (takes images & calibration, outputs point cloud & depth)
  - (Optional) Estimation (Object-Extrinsic if ground-truth geometry is known)
- Applications
  - (AI) Denoising
  - (AI) Hole_Filling
  - (Future) Shape/Material Optimization (regular gradient descent or with differentiable renderer if too many parameters)
