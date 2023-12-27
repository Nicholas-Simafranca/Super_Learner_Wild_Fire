# Modeling wildland fire burn severity in California using a spatial super learner approach

We develop a novel machine learning model to predict post-fire burn severity using pre-fire remotely sensed data. Hydrological, ecological, and topographical variables collected from four regions of California — the site of the Kincade fire (2019), the CZU Lightning Complex fire (2020), the Windy fire (2021), and the KNP Fire (2021) — are used as predictors of a burn severity response variable. 

## Table of Contents

- [Folder Structure](#folder-structure)
- [Usage](#usage)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## Folder Structure

These folders contain the complete end-to-end coding processes used in this project.  

### /Data Manipulation
This folder contains the data manipulation.

- Missing data is present due to factors such as cloud coverage; We removed variables with more than 28% missing pixels across each geographic region
- For the remaining variables, missing observations were imputed using K Nearest Neighbor imputation, with k=10 and distance defined using latitude and longitude
- Many of the variables are collected at different times within the two-week period leading up to the fire ignition

    - To resolve collinearity between subsequent measurements and harmonize variables across fires with a different number of replications of the variables, we convert the sequence of observations to estimates of the current value and the trend at the time of ignition for each pixel

### /Functions
This folder contains miscellaneous functions used over the duration of this project, such as:

- KNN Imputation method
- Main effects plot
- Mapping plot
- etc.

### /Methods
This folder includes 

### /Results

## Usage

Provide examples or instructions on how to use your project.

## Installation

Guide users on how to install or set up your project.

## Contributing

Explain how others can contribute to your project.

## License

Specify the project's license information.
