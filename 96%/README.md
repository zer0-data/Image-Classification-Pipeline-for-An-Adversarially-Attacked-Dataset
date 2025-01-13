# CLIP-based Image Classification Using Multiple Models

## Overview
This notebook implements a robust image classification system using an ensemble of CLIP models. The system uses multiple prompt templates and test-time augmentations to improve classification accuracy and reliability.

## Features
- Ensemble of CLIP models (ViT-Base and ViT-Large)
- Enhanced prompt engineering with contextual templates 
- Test-time augmentation for improved robustness
- Weighted model ensemble predictions
- Error handling with fallback predictions

## Implementation Details

### Model Ensemble
The system uses two CLIP variants:
- ViT-Base (32 patch size): 40% weight in ensemble
- ViT-Large (14 patch size): 60% weight in ensemble

### Prompt Engineering
Each class uses 6 different prompt templates with varying weights:
1. Natural habitat (1.2x weight)
2. High resolution (1.0x weight)
3. Close-up wildlife (1.1x weight)
4. Professional photograph (1.0x weight)
5. Clear detailed shot (1.1x weight)
6. Nature photograph (1.0x weight)

### Test-Time Augmentation
Four augmentation strategies are used:
1. Original image (1.0x weight)
2. Horizontal flip (0.8x weight)
3. 85% center crop (0.9x weight)
4. 95% center crop (0.9x weight)

### Output
The system generates a 'predictions.csv' file containing:
- image_id: Name of the input image
- class: Predicted class name

## Error Handling
- Robust error handling for image loading and processing
- Fallback to 'antelope' class if processing fails
- Progress bar with error reporting during batch processing

## Performance Notes
- CUDA acceleration enabled by default
- cuDNN benchmarking enabled for optimal performance
- Batch processing for memory efficiency
