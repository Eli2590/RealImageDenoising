# RealImageDenoising

- Download SIDD medium dataset with 320 image pairs and Flicker dataset in dedicated folders
- To create noisy and GT paired patches use command:  
  `python generate_rgb_images.py --use_gpu`
- To create cropped images dataset from sidd training dataset use command:  
  `python crop_rgb_images.py --use_gpu`  
  This will create 250 crops of size 64x64 for each image
- Training:  
  - To train denoiser on flickr dataset use:  
    `python train_denoiser.py --use_gpu --save_images --use_test_set`
  - To train denoiser on sidd dataset using pretraing model on flickr dataset use:  
    `python train_denoiser.py --dataset sidd_train --use_gpu --save_images --use_flickr`
  - To train denoiser on sidd dataset without pretraing model on flickr dataset use:  
    `python train_denoiser.py --dataset sidd_train --use_gpu --save_images`
- Testing:
  - To test denoiser trained on a model which was trained on flickr dataset and sidd dataset:  
    `python test_denoiser.py --use_gpu --save_images --use_flickr`
  - To test denoiser trained on a model which was trained only on sidd dataset:  
    `python test_denoiser.py --use_gpu --save_images`