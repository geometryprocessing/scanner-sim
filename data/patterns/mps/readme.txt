1) If the scene has very high-frequency interreflections, use higher-
frequencies (16 pixels) and more frequencies (8-15).

2) If the scene has defocus/subsurface scattering and no/weak
interreflections, use lower frequencies (32-64 pixels). In this case, a small
number of frequencies may be sufficient (5-8).

/usr/bin/gm convert Frame017.bmp -resize 1024x1080! -background white -extent 1920x1080 frame_17.png
