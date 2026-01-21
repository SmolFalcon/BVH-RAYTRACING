# BVH-RAYTRACING
A minimal, Bounding Volume Hierarchy accelerated GPU Raytracer for CUDA

Made for learning, this is a raytracing render engine implemented in CUDA. A .bin file is generated through a script inside a Cinema 4D scene, material implementation is minimal, but the code can be modyfied to represent different surface effects, like Oren-Nayar diffuse BRDF, Snell's law of refraction and mirror-like reflections with multi-sample roughness.

Most of the BVH implementation was referenced from *J. Bikker's Blog*[^1]

## Results
<img width="2048" height="1024" alt="mainScene" src="https://github.com/user-attachments/assets/98b43637-07b7-4a2d-8d0a-83defec2a01e" />
Main scene demostrating the features implemented, like rough multiple bounce reflections

<br/><br/><br/>
<img width="2048" height="1024" alt="birdSkull" src="https://github.com/user-attachments/assets/6018a297-aab2-4dad-8bfe-6df0196cdc29" />
Scene with complex translucent geometry

<br/><br/><br/>
<img width="1254" height="623" alt="Captura de pantalla 2026-01-19 144930" src="https://github.com/user-attachments/assets/ab120836-1f0b-4b33-847b-b9be322f7473" />
Scene showcasing shadow casting from a point light source with multi-sample softness


<br/><br/><br/>
<img width="391" height="259" alt="Captura de pantalla 2026-01-19 144406" src="https://github.com/user-attachments/assets/39255c96-7eee-4ebc-aacb-434005684895" /><br/>
Simple console status print during render

<br/><br/><br/>
## References
[^1]: https://jacco.ompf2.com/author/jbikker/
