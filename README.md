# Production-oriented Snow Water Equivalent Forecasting Workflow

## Project Goal

This project aims to develop a robust and scalable workflow for predicting software engineering outcomes in the context of the western United States. This project leverages machine learning and geospatial analysis techniques to address the unique challenges and characteristics of software development in this region.

We basically have the barebone workflow up and running, but the machine learning models and some steps need a lot of work to get it function correctly and up to our expection. Please refer to the `Key Problems to Solve` section to get more details about the issues that are puzzling right now.

Some preliminary results and machine learning models are obviously underestimating most time.

![2022_predicted_swe_2022-03-15](https://github.com/geo-smart/swe_forecasting_prod/assets/17322987/f0937113-e757-4022-877c-f46ee332bce4)

The current involved datasets that are used to create the `training.csv`:
- NASA SRTM 30M DEM
- [NSIDC AMSR](https://nsidc.org/data/au_dysno/versions/1)
- [gridMET Meteo Datasets](https://www.northwestknowledge.net/metdata/data/)
- SNOTEL ground observations

## Development Platform

We will have three development platforms for each team member:

- We will help create a similar Anaconda environment (including identical python environment to the workflow runtime environment) on everyone's local laptops to do some handy small experiments. This can be considered as prototyping.
- One AWS EC2 g5g.16xlarge instance (price: $2.744/hr), which will be always alive during the hackweek. Every team member will have access. It will have Geoweaver and all the dependencies preinstalled so ready to use when hackweek started. This is our staging server, which means it is used to fully test your code in a production-similar environment.
- One AWS EC2 p2.16xlarge instance (price: $14.40), which will be only alive when we are ready to roll in reality. We will only have one user credentials and will only share with administrators or pipeline managers. This will be our eventual operational environment. Because p2 instance has 16GPUs, our machine learning model will run at fully capacity in this server. 


## Tools and Software

We used GeoWeaver, a handy workflow management system, designed to help us create, manage, and execute the workflow. The foundation workflow skeleton is already created, and all the team members can build on top of it to save huge amount of time avoiding going back and forth to debug technical issues. Most steps are ready to plug and play usage. Geoweaver allows everyone's work to be automatically recorded, shared, utilized, and deployed seamlessly and effortlessly. 

<img width="1205" alt="image" src="https://github.com/geo-smart/swe_forecasting_prod/assets/17322987/899b9694-6863-45d4-9b76-9c90820ced05">

## Machine Learning Models


## Key Problems to Solve


## Expected Results


## How to contribute?


## Prerequisite Skills



