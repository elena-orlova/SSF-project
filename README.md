# Beyond Ensemble Averages: Leveraging Climate Model Ensembles for Subseasonal Forecasting

A supplementary code for our [paper](https://journals.ametsoc.org/view/journals/aies/3/4/AIES-D-23-0103.1.xml).


## Data 

TData include NCEP-NSCv2 and NASA-GMAO ensemble members, the ground truth data -- precipitation and 2 meter temperature, observational data: slp, rhum, hgt500; and principal components of SSTs. There are also files with the US mask, an observational climatology, a model climatology and the 33rd and 66th percentile values.

All data was preprocessed: the ensemble members, the ground truth and the climate data are on the same grid. There are two directories:

- train_val -- data for training and validation.
- test -- data for test. 

Usually, all variables have the following shape (t, 64, 128), where t = 312 for train and validation, t = 117 for NCEP data and t = 85 for NASA data for test. File ```utils_data.py``` includes functions to work with the data.

Data used for the experiments can be found [here](https://uchicago.box.com/s/xzv588kzyywykdfmsucwntpnd06zf79w). 

## Prerequisites

See ```requirements.txt``` file.
- Python 3.3+
- [smp](https://github.com/qubvel/segmentation_models.pytorch) package

## Basic usage

There are a few Jupyter notebooks to demonstrate how to work with data and train/evaluate models:

- Regression example + tercile classification of temperature [example](https://github.com/elena-orlova/SSF_project/tree/master/regression&tercile_classification/regression_baselines_RF_tercile_tmp.ipynb)
    - Regression with U-Net + tercile classification of temperature [example](https://github.com/elena-orlova/SSF_project/tree/master/regression&tercile_classification/regression_UNET_tercile_tmp.ipynb)
- Model stacking [example](https://github.com/elena-orlova/SSF_project/tree/master/regression&tercile_classification/stacking_models.ipynb)
- Tercile classification of temperature [example](https://github.com/elena-orlova/SSF_project/tree/master/regression&tercile_classification/tercile_classification_precip.ipynb)
<!-- - Quantile regression [example](?) -->


## Citation

```
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
