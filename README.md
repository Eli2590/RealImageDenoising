# RealImageDenoising

- Download SIDD medium dataset with 320 image pairs and Flicker dataset in dedicated folders
- To create noisy and GT paired patches use command:  
  `python generate_rgb_images.py --use_gpu`
- To create cropped images dataset from sidd training dataset use command:  
  `python crop_rgb_images.py --use_gpu`  
  This will create 250 crops of size 64x64 for each image