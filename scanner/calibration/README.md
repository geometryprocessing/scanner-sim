All the calibration data (except projector vignetting - TODO) is stored in form of *.json files containing a Python dictionary data structure with all the relevant parameter:value pairs ([example](calibration/camera/camera_geometry.json)).
Camera and projector vignetting are also saved as pre-evaluated 8-bit *.png images ([camera](camera/vignetting/inverted_softbox_smooth.png), [projector](projector/vignetting/White_200ms_vignetting.png)).

Use load_calibration() function from [utils.py](utils.py) to load calibration data in numpy ready format.
Also, please see the [chart](parameters_flow.pdf) for calibration parameters flow / usage during calibration procedure. More details can be found in our [supplementary material](../../scanner_sim_technical.pdf).
