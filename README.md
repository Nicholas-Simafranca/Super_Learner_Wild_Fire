# Modeling wildland fire burn severity in California using a spatial super learner approach

We develop a novel machine learning model to predict post-fire burn severity using pre-fire remotely sensed data. Hydrological, ecological, and topographical variables collected from four regions of California — the site of the Kincade fire (2019), the CZU Lightning Complex fire (2020), the Windy fire (2021), and the KNP Fire (2021) — are used as predictors of a burn severity response variable. We first analyze the four fires separately to obtain within-fire predictions. We also combined data from all four fires to form within-fire predictions. 

## Folder Structure

These folders contain the complete end-to-end coding processes used in this project.  

### /Data Manipulation
This folder contains the data manipulation. Codes are organized according to three different file names in which reproducible processes were carried out: 

- FIRE_EDA.Rmd
     - Removed variables with more than 28% missing pixels
     - Missing observations were imputed using K Nearest Neighbor Imputation, with k=10 and distance defined used latitude & longitude
- FIRE_Grouping.Rmd
     - Many of the variables are collected at different times within the two-week period leading up to fire ignition
          - Convert the sequence of observations to estimates of the current value and the trend at the time of ignition for each pixel 
- FIRE_final_grouping.Rmd
     - Standardize all covariates in data set before subsequent analysis 

### /Functions
This folder contains miscellaneous functions used over the duration of this project, such as:

- KNN Imputation method
- Main effects plot
- Mapping plot
- etc.

### /Methods
This folder includes the SL algorithm, an ensemble learning method that combines base learners to achieve superior predictive accuracy. Codes are broken up across: 

- four separate fires
- combined data from all four fires 

### /Results
This folder includes the results obtained for both within-fire predictions and combined-fire predictions. Four different types of models are compared across these predictions. These models differ by whether the original covariates or base learners are included in the mean and whether the spatial random effects are included in the covariance. The codes for such analyses are organized by: 

1. FIRE_LR_Ind.Rmd
2. FIRE_LR_spatial.Rmd
3. FIRE_SL_Ind.Rmd
4. FIRE_SL_spatial.Rmd

