# Aerial View Image Translation

# Overview

This project provides a script for testing a trained image translation model using PyTorch. The script takes input SAR or RGB images, performs inference using the trained model, and saves the translated images (EO, IR, RGB) to a specified results directory.

## Requirements
- Python 3.9
- PyTorch
- torchvision
- numpy
- Pillow

## Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/hamzashafiq28/aerial-view-image-translation.git
    ```
2. Install the required dependencies:
     ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Navigate to the directory containing the script.
2. Download the input data and pretrained weights using this link
   https://drive.google.com/file/d/1wqMZE3eKIsNip7to4WcmW0F5-UWqNz7M/view?usp=sharing
4. Run the script with the desired command-line arguments:
    ```css
    python main.py [--model_path MODEL_PATH] [--data_dir DATA_DIR] [--results_dir RESULTS_DIR] [--type TYPE]
    ```
    - `--model_path`: Path to the trained model file (default: 'pretrained/sar2ir/SAR_IR.pth').
    - `--data_dir`: Path to the directory containing the input images (default: 'input/sar2ir').
    - `--results_dir`: Directory to save the resulting translated images (default: 'results/sar2ir').
    - `--type`: Type of input data ('SAREO', 'SARRGB', 'SARIR', 'RGBIR') (default: 'SARIR').


## Notes
- Ensure that the directory paths provided exist and contain the necessary data.
- The script automatically selects the appropriate input domain based on the specified data type.

## Example

### SAR to EO

```css
python main.py --model_path pretrained/sar2eo/SAR_EO.pth --data_dir input/sar2eo --results_dir results/sar2eo --type SAREO
```

### SAR to IR

```css
python main.py --model_path pretrained/sar2ir/SAR_IR.pth --data_dir input/sar2ir --results_dir results/sar2ir --type SARIR
```

### SAR to RGB

```css
python main.py --model_path pretrained/sar2rgb/SAR_RGB.pth --data_dir input/sar2rgb --results_dir results/sar2rgb --type SARRGB
```

### RGB to IR

```css
python main.py --model_path pretrained/rgb2ir/RGB_IR.pth --data_dir input/rgb2ir --results_dir results/rgb2ir --type RGBIR
