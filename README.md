# Solvent Swap Optimisation

### Timeline for the project

A manim project shows the flow of data within this project. The idea is this can be run to provide a systematic update on progress.

## Prerequisities 
- Install the prerequisites within [requirements.txt](requirements.txt)
- Run all files from the root directory

## Extract, Transform & Load

ETL must occur on the provided assets, alongside processing of Bao et al. data as this provided a stable to base the future algorithms on. 

- Run the [main.py](src/1-ETL/main.py) within the [1-ETL](src/1-ETL/) folder

TODOS
- Find out about the failed entries
- redo the dataPreparationPubChem script to use sql instead
- Use MW to convert to mol per mol
- Find Density values? - ASK THIS QUESTION
- saturation enum - ASK THIS QUESTION
- vol fraction - TEMPERATURE DEPENDENCE + DENSITY - ASK THIS QUESTION
- NaCl as a solvent??? 
- UNITS FOR solubility non existent?


## Feature Engineering

## Variational Autoencoder

## VAE Validation

## Bayesian Optimisation of Latent Space
