# Open-Unmix Music Source Separation Project
## Overview

This project conducts an **experimental analysis of Open-Unmix hyperparameters**, a deep neural network for music source separation. Rather than implementing new models, this study systematically investigates how different hyperparameter configurations affect model performance, training time, and separation quality. The analysis focuses on understanding the impact of batch size, hidden size, samples per track, and LSTM directionality on the existing Open-Unmix architecture.

## Features

- **Hyperparameter Analysis**: Systematic study of batch size, hidden size, and samples per track effects
- **Performance Comparison**: Evaluation across multiple model configurations using existing Open-Unmix architecture
- **Real-time Processing Investigation**: Analysis of unidirectional vs bidirectional LSTM for real-time applications

## Installation

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended)
- FFmpeg

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sigsep/open-unmix-pytorch.git
   cd open-unmix-pytorch/
   ```

2. **Install dependencies**:
   ```bash
   # Install Open-Unmix requirements from the main repository
   pip install -r scripts/requirements.txt
   
   # Additional packages required for this experimental study
   pip install musdb museval youtube_dl GitPython
   ```

3. **Install FFmpeg**:
   ```bash
   sudo apt-get install ffmpeg
   ```

4. **Prepare MUSDB18 dataset** (if training from scratch):
   ```bash
   # Convert MUSDB18 to WAV format
   musdbconvert /path/to/musdb18/ ./musdb-wav-root
   ```

## Usage

### Running the Experiment

This project focuses on **hyperparameter analysis** using existing Open-Unmix models. To reproduce the experimental results:

1. **Follow the step-by-step instructions** in the main notebook: `Open_Unmix_635_FinalCode.ipynb`
2. **For detailed results and analysis**, refer to the main report: `main_report.pdf`

The notebook contains all necessary code cells to:
- Set up the environment and dependencies
- Train models with different hyperparameter configurations
- Evaluate model performance using various metrics
- Generate comparison results across different model variants

## Experimental Study Results

This project includes a comprehensive analysis of different model configurations:

### Training Parameters and Performance

| Model | Batch Size | Workers | Hidden Size | Samples/Track | Epochs | Final Loss | Training Time |
|-------|------------|---------|-------------|---------------|---------|------------|---------------|
| open-unmix | 8 | 4 | 256 | 16 | 1 | 9.536 | 7:50 |
| open-unmix2 | 4 | 4 | 256 | 16 | 1 | 8.55 | 7:43 |
| open-unmix3 | 16 | 4 | 512 | 16 | 1 | 11.2 | 7:42 |
| open-unmix4 | 16 | 4 | 512 | 16 | 4 | 3.2 | 31:09:00 |
| open-unmix5 | 32 | 4 | 512 | 32 | 5 | 2.26 | 1:18:00 |
| open-unmix6 | 32 | 4 | 512 | 32 | 1 | 9.56 | 20:00 |

### Audio Separation Quality Metrics

| Model | SDR | SIR | ISR | SAR |
|-------|-----|-----|-----|-----|
| UMX (Original) | 7.537 | 12.091 | 14.586 | 8.679 |
| open-unmix | 2.326 | 3.028 | 6.588 | 5.347 |
| open-unmix2 | 2.746 | 2.954 | 7.706 | 6.288 |
| open-unmix3 | 1.934 | 2.849 | 5.579 | 4.07 |
| open-unmix4 | 2.767 | 3.052 | 9.831 | 7.201 |
| open-unmix5 | 2.898 | 3.628 | 9.272 | 6.63 |

### Model Performance Analysis

- **open-unmix2**: With the smallest batch size (4) and hidden size (256), achieved loss reduction to 8.55 in 7:43 for one epoch
- **open-unmix3**: Despite increasing batch size to 16 and doubling hidden size to 512, showed increased loss (11.2) but maintained similar training time (7:42)
- **open-unmix4**: Same parameters as open-unmix3 but trained for 4 epochs, drastically reduced loss to 3.2 with significantly longer training time (31+ minutes)
- **open-unmix5**: Achieved the lowest loss (2.26) with batch size 32 and 32 samples per track over 5 epochs (1:18:00 training time)

### Key Findings

1. **Batch Size Impact**:
   - No straightforward relationship between batch size and loss reduction
   - Training time not directly correlated with batch size (number of batches reduced proportionally)
   - Comparison between open-unmix2 and open-unmix3 illustrates this complexity

2. **Hidden Size Optimization**:
   - 256 hidden units optimal for processing 16kHz signals
   - Performance begins to decrease above 256 due to overfitting from capturing subtle frequency patterns
   - ReLU activation layers make the model more robust to increased hidden size

3. **Real-time Processing**:
   - Unidirectional LSTM can process signals in real-time but heavily depends on machine characteristics
   - Model needs to process audio batches faster than playback rate for real-time separation
   - 16kHz frequency cropping significantly improves processing speed
   - GPU presence or multiple cores significantly boost separation time

4. **Training Duration vs Performance**:
   - Extended training (multiple epochs) significantly improves loss reduction
   - Trade-off between training time investment and model performance gains

## Research Questions Addressed

1. **How do different hyperparameters (batch size, hidden size, samples per track) affect model performance and training time?**
   - Systematic comparison across 5 different configurations
   - Analysis of loss convergence and computational efficiency

2. **Can unidirectional LSTM replace bidirectional LSTM for real-time processing without significant quality loss?**
   - Investigation of architectural modifications for real-time applications
   - Trade-off analysis between processing speed and separation quality

## Performance Metrics

The project evaluates model quality using standard source separation metrics:
- **SDR (Source-to-Distortion Ratio)**: Overall separation quality - higher is better
- **SIR (Source-to-Interference Ratio)**: Interference from other sources - higher is better  
- **ISR (Image-to-Spatial distortion Ratio)**: Spatial distortion measure - higher is better
- **SAR (Source-to-Artifact Ratio)**: Artifacts introduced by separation - higher is better

## File Structure

```
project/
├── Open_Unmix_635_FinalCode.ipynb  # Main experimental notebook (follow step-by-step)
├── main_report.pdf            # Detailed experimental analysis and results
├── open-unmix-study.pdf            # Summary experimental analysis  
```

## Hardware Requirements

### Minimum Requirements
- 8GB RAM
- CUDA-compatible GPU with 4GB VRAM
- 10GB free disk space

### Recommended for Training
- 16GB+ RAM
- CUDA-compatible GPU with 8GB+ VRAM
- SSD storage for dataset
