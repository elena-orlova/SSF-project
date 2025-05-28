# Beyond Ensemble Averages: Leveraging Climate Model Ensembles for Subseasonal Forecasting

[![Paper](https://img.shields.io/badge/Paper-AIES-blue)](https://journals.ametsoc.org/view/journals/aies/3/4/AIES-D-23-0103.1.xml)
[![Python](https://img.shields.io/badge/Python-3.3+-green)](https://www.python.org/)
[![Data](https://img.shields.io/badge/Data-Available-orange)](https://uchicago.box.com/s/zx71j8brfjhmrcjop3a5kwf9k3ydfozo)

This repository contains the supplementary code for our paper on leveraging climate model ensembles for improved subseasonal forecasting performance beyond traditional ensemble averaging approaches.

## 📄 Paper

**[Beyond Ensemble Averages: Leveraging Climate Model Ensembles for Subseasonal Forecasting](https://journals.ametsoc.org/view/journals/aies/3/4/AIES-D-23-0103.1.xml)**  
*Artificial Intelligence for the Earth Systems, Vol. 3, No. 4 (2024)*

## 🗂️ Data Description

The dataset includes comprehensive climate model outputs and observational data:

### Data Sources
- **NCEP-NSCv2** ensemble members
- **NASA-GMAO** ensemble members
- **Ground truth data**: precipitation and 2-meter temperature
- **Observational data**: sea level pressure (slp), relative humidity (rhum), 500mb geopotential height (hgt500)
- **Principal components** of sea surface temperatures (SSTs)

### Additional Files
- US mask for regional analysis
- Observational climatology
- Model climatology
- 33rd and 66th percentile threshold values

### Data Structure
All data has been preprocessed and regridded to ensure consistency across ensemble members, ground truth, and climate variables.

```
train_val/     # Training and validation data
test/          # Test data
```

**Data Dimensions**: `(t, 64, 128)` where:
- Training/validation: `t = 312`
- NCEP test data: `t = 117`
- NASA test data: `t = 85`

### Data Access
Download the complete dataset: **[SSF Data](https://uchicago.box.com/s/zx71j8brfjhmrcjop3a5kwf9k3ydfozo)**

## Prerequisites

### System Requirements
- Python 3.3 or higher
- Additional dependencies listed in `requirements.txt`
- [Segmentation Models PyTorch (smp)](https://github.com/qubvel/segmentation_models.pytorch)


It can be installed as
```bash
pip install -r requirements.txt
```

## Getting Started

### Data Utilities
Use `utils_data.py` for data loading and preprocessing functions.

### Example Notebooks

We provide several Jupyter notebooks demonstrating different modeling approaches:

#### 1. Regression and Tercile Classification
- **[Regression Baselines + Temperature Tercile Classification](regression&tercile_classification/regression_baselines_RF_tercile_tmp.ipynb)**
  - Random Forest regression with tercile classification for temperature
  
- **[U-Net Regression + Temperature Tercile Classification](regression&tercile_classification/regression_UNET_tercile_tmp.ipynb)**
  - Deep learning approach using U-Net architecture

#### 2. Advanced Modeling Techniques
- **[Model Stacking](regression&tercile_classification/stacking_models.ipynb)**
  - Ensemble methods combining multiple model predictions
  
- **[Precipitation Tercile Classification](regression&tercile_classification/tercile_classification_precip.ipynb)**
  - Categorical forecasting for precipitation events


## 🔬 Methodology

Our approach goes beyond simple ensemble averaging by:
1. Leveraging individual ensemble members rather than just ensemble means
2. Using deep learning to capture spatial patterns and relationships
3. Implementing tercile-based classification for categorical forecasting
4. Combining multiple models through stacking techniques

## 📈 Results

The methods demonstrated in this repository show improved subseasonal forecasting performance compared to traditional ensemble averaging, particularly for extreme weather events and spatial pattern prediction.

## 📚 Citation

```bibtex
@article{orlova2024beyond,
  title={Beyond ensemble averages: Leveraging climate model ensembles for subseasonal forecasting},
  author={Orlova, Elena and Liu, Haokun and Rossellini, Raphael and Cash, Benjamin A and Willett, Rebecca},
  journal={Artificial Intelligence for the Earth Systems},
  volume={3},
  number={4},
  pages={e230103},
  year={2024},
  publisher={American Meteorological Society}
}
```
