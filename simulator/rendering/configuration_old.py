# Write render configuration
import numpy as np
import json
import os

def generate_render_parameters(obj_path, result_path, typ="calib", size="medium", cpu_count=12, samples=32, patterns="patterns/gray/predistorted"): 
    #typ shapes, pawn, lplan, plane, calib, center, rook
    #size medium, small, large

    cam_w = 6464
    cam_h = 4852

    if size == "large":
        cam_w = 6464
        cam_h = 4852
    if size == "medium":
        cam_w = int(cam_w/2)
        cam_h = int(cam_h/2)
    if size == "small":
        cam_w = int(cam_w/4)
        cam_h = int(cam_h/4)

    cam_wa = int(cam_h * 1.3331977988173698) # Account for pixel aspect ratio
    
    # Projector rotation and translation old
    #r = np.array([[0.8787054478263335,  -0.01630932436494734,  0.47708567563812],
    #             [0.012209582502145412, 0.9998570945107125, 0.011692589607071084],
    #             [-0.47720819571277884, -0.004449325269635517, 0.8787790060357604]])
    #t = np.array([[0.2256677886438659, 0.07394777683137411, 0.4140179434986635, 1.0]])
    
    # Projector rotation and translation new
    r = np.array([[0.8782723416784038, -0.01631591654536631, 0.4778822916890913],
                  [0.01195247503124907, 0.9998544993767309, 0.012170390985267683],
                  [-0.47801133060144135, -0.004977041630482834, 0.87833956809041]])
    t = np.array([[0.22609041436682526, 0.07406257781319086, 0.4142945658297809, 1.0]])
    
    r = np.vstack([r.T, np.zeros(3)])
    trans_pro = np.hstack([r, t.T]).reshape(-1)

    pars = {
      # Object Rotation parameters
      "rot_x": 0,
      "rot_y": 0,
      "rot_z": 0,
      "rot_range": [[-45, 45], [-45, 45], [-45, 45], 30],
      "rot_type": "Fixed", #"Turntable", #"Fixed", "Random"
      # Object Translation parameters
      "trans_x": 0.0,
      "trans_y": 0.0,
      "trans_z": 0.0,
      "trans_range": [[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1], 5],
      "trans_type": "Fixed",
      # Camera parameters
      "cam_aperture": 0.00016,
      "cam_focus": 0.81,
      "cam_fovy": 18.728257607815436,
      "cam_tx": 0.0,
      "cam_ty": 0.0,
      "cam_tz": 1.0,
      "cam_ux": 0.0,
      "cam_uy": -1.0,
      "cam_uz": 0.0,
      "cam_x": 0.0,
      "cam_y": 0.0,
      "cam_z": 0.0,
      "cam_height": cam_h,
      "cam_width": cam_wa,
      "cam_crop_offset_x": int((cam_wa-cam_w)/2),
      "cam_crop_offset_y": 0,
      "cam_crop_width": cam_w,
      "cam_crop_height": cam_h,
      # Global parameters
      "cpu_count": cpu_count,
      "samples": samples,
      "result_path": result_path,
      "result_overwrite": True,
      "scene_path": "scenes/scene_sls.xml",
      "root_path": "data",
      "render_ambient": True,
      # Pattern parameters
      "pattern_path": patterns,#mps/64-05", gray/default
      "pattern_numbered": True, # If patterns have numbered names
      "pattern_colored": False, # Colored patterns/rendering
      "pattern_calibrate": True, # Calibrate/predistort patterns
      "pattern_overwrite": True, # Overwrite possible existing patterns
      "pattern_quadrify": True, # Pad patterns for quadratic spotlight
      "pattern_pad_below": True,
      "pattern_flip_ud": True,
      "pattern_width": 1920,
      "pattern_height": 1080,
      # Object parameters
      "shape_type": obj_path[-3:], # ply/obj
      "object_path": obj_path,
      "scale": 0.001, # 0.001 for calib objects
      "obj_rx": 0.0,
      "obj_ry": 0.0,
      "obj_rz": 0.0,
      "obj_ttx": 0.0,
      "obj_tty": 0.0,
      "obj_ttz": 0.0,
      "obj_x": 0.0,
      "obj_y": 0.0,
      "obj_z": 0.0,
      "obj_trans": None,#trans.tolist(),
      # Projector parameters
      "pro_aperture": 0.0002835,
      "pro_beamwidth": 21.33245815404095,
      "pro_cutoff": 21.33245815404095,
      "pro_focus": 0.49, #0.49 before
      "pro_intensity": 0.5,
      "pro_offset_x": 0.5,#+0.00045641259698767686,
      "pro_offset_y": 0.0,#-0.007074395253308992,
      "pro_scale_x": 0.5,
      "pro_scale_y": 0.5,
      "pro_tx": 0.0,
      "pro_ty": 0.0,
      "pro_tz": 0.0,
      "pro_ux": 0.0,
      "pro_uy": 0.0,
      "pro_uz": 0.0,
      "pro_x": 0.0,
      "pro_y": 0.0,
      "pro_z": 0.0,
      "pro_ox": 0.0,
      "pro_oy": 0.0,
      "pro_oz": 0.0,
      "proj_trans": trans_pro.tolist(),
      # Lighting/Material parameters
      "const_radiance": 0.01,
      "mat_ior": 1.0634878252015734,
      "mat_alpha": 0.22150848706292028,
      "mat_diff": 0.9112623909906508
    }
    
    if typ == "pawn":
        r = np.array([[9.99985088e-01, -5.46014155e-03, -1.06692981e-04],
                      [0.00000000e+00, -1.95366030e-02,  9.99809142e-01],
                      [-5.46118386e-03, -9.99794233e-01, -1.95363117e-02]])
        t = np.array([[-.00948753506, .07090617493, 0.86120278452, 1.0]])
        
    if typ == "shapes":
        r = np.array([[0.00473251, 0.99973483, 0.02252548],
                     [0.70316257, -0.0193432,   0.71076595],
                     [0.71150094,  0.01246195, -0.70257463]])
        t = np.array([[-.07613365619, -0.10450656288, 0.81651577885, 1.0]])
        #pars["scene_path"] = "scenes/scene_sls_tex.xml"
    
    if typ == "center":
        r = np.array([[1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0],
                      [0, 0.0, -1.0]])
        t = np.array([[0.0, -0.02, 0.858, 1.0]])
        
    if typ == "rook":
        t = np.array([[-.00965021966, .07090294057, 0.86141378352, 1.0]])
        r = np.array([[ 9.99985088e-01, -5.46014155e-03, -1.06692981e-04],
                     [ 0.00000000e+00, -1.95366030e-02,  9.99809142e-01],
                     [-5.46118386e-03, -9.99794233e-01, -1.95363117e-02]])

    if typ == "plane":
        t = np.array([[ -.0083332819, -.00283606196, 0.83846561563, 1.0]]) 
        r = np.array([[0.90031236, -0.01464571,  0.43499787],
                      [0.01133496, 0.99988369, 0.01020466],
                      [-0.43509673, -0.0042567, 0.90037365]])

    if typ == "lplane":
        #t = np.array([[-.00795539039, -.02476862133, 0.84448289208, 1.0]])
        #        r = np.array([[0.72965066, -0.46024329,  0.50575293],
        #                       [0.38697703, -0.88768783, 0.24951773],
        #                       [0.56378958, 0.01365399, -0.82580559]])
        # Clear plane
        t = np.array([[0.007459225901051556,-0.02054595703481133, 0.8180658161193445, 1.0]])
        r = np.array([[0.8420082513171478, -0.031627971059968396, 0.5385366989913214],
                       [0.014386377983397893, -0.9992413108596305, 0.03619163991926788],
                       [0.5392727851860859, 0.022726066931101685, -0.8418244407472882]])
        # Charuco plane
        t = np.array([[-0.16342898201514, 0.10846442812345023, 0.710381808565605, 1.0]])
        r = np.array([[0.8412727166932105, -0.0034109062995231303, 0.5406002052032919],
                      [-0.01386375902532436, -0.9997873439322098, 0.015266404248050983],
                      [0.5404331710149861, -0.020337960349873955,-0.8411411029283338]])
        r = r.T
        pars["scale"] = 0.1
        #pars["scale"] = 0.001
        
    colored = False
        
    if typ == "vase":
        trans = np.array([[ 5.98357629e+01, 1.73780698e+00, -5.15957625e+01, -1.13917137e+01],
                         [-2.11396908e-01, -7.89745329e+01, -2.90511434e+00,  3.18578817e+01],
                         [-5.16245870e+01,  2.33760665e+00, -5.97904575e+01,  8.60359595e+02],
                         [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        colored = True
 
    if typ == "sculpture":
        trans = np.array([[5.93805928e+01, -7.04532055e-01, -4.45788431e+01, -1.34888086e+01],
                         [2.85445598e-01, -7.42383296e+01,  1.55349953e+00,  7.14825672e+01],
                         [-4.45834963e+01, -1.41367413e+00, -5.93644490e+01,  8.52635829e+02],
                         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        colored = True
        
    if typ == "dodo":
        trans = np.array([[2.40875072e+00, -4.40313574e-02, -1.03494969e+01, -7.01796345e+00],
                 [ 2.11953027e-01, -1.06236644e+01,  9.45279153e-02,  5.98905411e+01],
                 [-1.03474200e+01, -2.27861475e-01, -2.40729791e+00,  8.61709585e+02],
                 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        colored = True
                         
    if typ == "vessel":
        trans = np.array([[-441.30169547,  123.25346392, -295.41965328,  -10.90231931],
                         [-131.66575803, -528.49641954  ,-23.81253936  , 31.26875861],
                         [-291.767572,     52.07199344,  457.57137778,  861.65219967],
                         [   0.,            0.  ,          0.  ,          1.        ]])
        colored = True
    
    if typ == "radio":
        trans = np.array([[3.25797368, -1.59308514e-02, 3.07816167e-02, -9.46238943e+00],
                         [-1.65533222e-02, -3.25744425e+00,  6.61572727e-02,  3.60467132e+01],
                         [ 3.04513953e-02, -6.63099177e-02, -3.25734087e+00,  8.38955738e+02],
                         [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        colored = True
                         
    if typ == "chair":
        trans = np.array([[8.90331963e-01, -2.30152399e-03, -5.09032053e-01, -9.04789632e+00],
                           [9.57553323e-03, -1.02531018e+00,  2.13840743e-02,  7.11104463e+01],
                           [-5.08947185e-01, -2.33167849e-02, -8.90078100e-01,  8.62351044e+02],
                           [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        colored = True
                         
    if typ == "house":   
        trans = np.array([[ 1.27994214e+01, -5.73438996e-02,  1.05386119e+01, -1.07580296e+01],
                          [-2.63636666e-01, -1.65761399e+01,  2.29997678e-01, -4.80942864e+00],
                          [ 1.05354698e+01, -3.45130266e-01, -1.27974832e+01,  8.63149900e+02],
                          [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        colored = True

        
        
    if typ == "calib":
        pars['scene_path'] = "scenes/scene_sls_cal.xml"
        pars['object_path'] = "objects/calibration/cb10_8_3_ms_c.ply" 
        pars["rot_range"] = [[-15, 15], [-15, 15], [-15, 15], 1]
        pars["rot_type"] = "Random" #"Turntable", #"Fixed", "Random"
        pars["trans_range"] = [[-0.02, 0.02], [-0.01, 0.01], [-0.03, 0.1], 1]
        pars["trans_type"] = "Random"
        pars["render_ambient"] = False
        pars["const_radiance"] = 0.05
        pars["pro_intensity"] = 0.5
        pars["samples"] = 32
        r = np.array([[1, 0, 0.0], [0, 1, 0.0], [0, 0, 1.0]])
        t = np.array([[0.0, -0.02, 0.91, 1.0]])
        
    #pars['scene_path'] = "scenes/scene_sls_cal.xml"
    if not colored:
        r = np.vstack([r.T, np.zeros(3)])
        trans = np.hstack([r, t.T]).reshape(-1)
    
    else:
        trans[:3, :] = trans[:3, :] / 100
        trans[:3, 3] = trans[:3, 3] / 10
        trans = trans.reshape(-1)
        pars["scene_path"] = "scenes/scene_sls_tex.xml"
#         pars["pro_intensity"] = 20.0
#         pars["const_radiance"] = 0.1
        
    pars["obj_trans"] = trans.tolist()
    

    return pars
