# Beyond Ensemble Averages: Leveraging Climate Model Ensembles for Subseasonal Forecasting

Supplementary code for [the paper](?).
If you use our code/data/models in your research, we kindly ask you to cite our work
```
@article{?
}
```

## Data 

Data used for the experiemnts can be found [here](https://uchicago.box.com/s/xzv588kzyywykdfmsucwntpnd06zf79w). It includes NCEP-NSCv2 and NASA-GMAO ensemble members, ground truth data -- precipitation and 2 meter temperature, obseravtional data: slp, rhum, hgt500; principal components of SSTs. There are laso files with the US mask, observational climatology, the model climatology and 33rd and 66th percentile values.

All data was preprocessed: ensemble members, ground truth and climate data are on the same grid. There are two directories:

- train_val -- data for training and validation.
- test -- data for test. 

Usually, all variables have the following shape (t, 64, 128), where t = 312 for train and validation, t = 117 for NCEP data and t = 85 for NASA data for test. File ```utils_data.txt``` includes functions to work with the data.

## Prerequisites

See ```requirements.txt``` file.
- Python 3.3+
- [smp](https://github.com/qubvel/segmentation_models.pytorch) package

## Basic usage

There are a few Jupyter notebooks to demonstrate how to work with data and train/evaluate models.

- Regression example + tercile classification of temperature [example](https://github.com/elena-orlova/SSF_project/tree/master/regression&tercile_classification/regression_baselines_RF_tercile_tmp.ipynb)
    - Regression with U-Net + tercile classification of temperature [example](https://github.com/elena-orlova/SSF_project/tree/master/regression&tercile_classification/regression_UNET_tercile_tmp.ipynb)
- Model stacking [example](https://github.com/elena-orlova/SSF_project/tree/master/regression&tercile_classification/stacking_models.ipynb)
- Tercile classification of temperature [example](https://github.com/elena-orlova/SSF_project/tree/master/regression&tercile_classification/tercile_classification_precip.ipynb)
<!-- - Quantile regression [example](?) -->

