# Solvent Swap Optimisation

## Prerequisities 
- Install the prerequisites within [requirements.txt](requirements.txt)
- Run all files from the root directory

## Extract, Transform & Load

ETL must occur on the provided assets, alongside processing of Bao et al. data as this provided a stable to base the future algorithms on. 

- Run the [main.py](src/0-ETL/main.py) contained in the [1-ETL](src/0-ETL/) folder
- This will generate a db/MasterDatabase.db file (note: this process will take a couple of minutes...)
    - Alternatively, the database can be obtained from the [MasterDatabase.db](/docs/src/db/MasterDatabase.db) file and you can manually create a db/MasterDatabase.db file in the root directory
- Exploratory Data Analysis python scripts can be located in the [EDA folder](/src/1.1-exploratory-data-visualisation/)
    - This allows viewing of the database in Plotly Dash or Seaborn charts


## Feature Engineering

WIP...

## Variational Autoencoder

WIP...

- Predict LogS instead of S

## VAE Validation - comparison to previous methods

WIP...

## Bayesian Optimisation of Latent Space

WIP...

## Docs folder

Contains the code for a Nextjs web app to load the documentation and potential examples. WIP...

