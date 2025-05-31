# Open-Unmix Music Source Separation Project

![Open-Unmix](https://sisec18.unmix.app/static/img/hero_header.4f28952.svg)

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

### Key Findings

1. **Batch Size Impact**:
   - No straightforward relationship between batch size and loss reduction
   - Training time not directly correlated with batch size

2. **Hidden Size Optimization**:
   - 256 hidden units optimal for 16kHz signals
   - Larger hidden sizes may lead to overfitting
   - ReLU activation helps maintain robustness

3. **Real-time Processing**:
   - Unidirectional LSTM shows promise for real-time applications
   - 16kHz frequency cropping improves processing speed
   - Performance depends heavily on hardware capabilities

## Research Questions Addressed

1. **How do different hyperparameters (batch size, hidden size, samples per track) affect model performance and training time?**
   - Systematic comparison across 5 different configurations
   - Analysis of loss convergence and computational efficiency

2. **Can unidirectional LSTM replace bidirectional LSTM for real-time processing without significant quality loss?**
   - Investigation of architectural modifications for real-time applications
   - Trade-off analysis between processing speed and separation quality

## Performance Metrics

The project evaluates model quality using:
- **SDR (Source-to-Distortion Ratio)**: Overall separation quality
- **SIR (Source-to-Interference Ratio)**: Interference from other sources
- **SAR (Source-to-Artifact Ratio)**: Artifacts introduced by separation

## File Structure

```
project/
├── Open_Unmix_635_FinalCode.ipynb  # Main experimental notebook (follow step-by-step)
├── open-unmix-study.pdf            # summary experimental analysis and results
├── main_report.pdf                 # Detailed experimental analysis and results
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

## Applications

- **Music Production**: Stem extraction for remixing and mastering
- **Audio Research**: Source separation algorithm development
- **Commercial Systems**: Integration with music streaming platforms
- **Real-time Processing**: Live audio separation applications

