# Production-oriented Snow Water Equivalent Forecasting Workflow

## Project Goal

This project aims to develop a robust and scalable workflow for predicting snow water equivalent in the context of the western United States. This project leverages machine learning and geospatial analysis techniques to address the changing patterns of snowpacks in this region.

We basically have the barebone workflow up and running, but the machine learning models and some steps need work to get it function correctly and up to our expection. Please refer to the `Key Problems to Solve` section to get more details about the issues that are puzzling right now.

Some preliminary results and machine learning models are obviously underestimating most time.

![2022_predicted_swe_2022-03-15](https://github.com/geo-smart/swe_forecasting_prod/assets/17322987/f0937113-e757-4022-877c-f46ee332bce4)

The current datasets that are used to create the `training.csv`:
- NASA SRTM 30M DEM
- [NSIDC AMSR](https://nsidc.org/data/au_dysno/versions/1)
- [gridMET Meteo Datasets](https://www.northwestknowledge.net/metdata/data/)
- SNOTEL ground observations

## Development Platform

We will have three development platforms for each team member:

- We will help create a similar Anaconda environment (including identical python environment to the workflow runtime environment) on everyone's local laptops to do some handy small experiments. This can be considered as prototyping.
- One AWS EC2 g5g.16xlarge instance (price: $2.744/hr), which will be always alive during the hackweek. Every team member will have access. It will have Geoweaver and all the dependencies preinstalled so ready to use when hackweek started. This is our staging server, which means it is used to fully test your code in a production-similar environment.
- (prod only) One AWS EC2 p2.16xlarge instance (price: $14.40), which will be only alive when we are ready to roll in reality. We will only have one user credentials and will only share with administrators or pipeline managers. This will be our eventual operational environment. Because p2 instance has 16GPUs, our machine learning model will run at fully capacity in this server. 


## Tools and Software

We used GeoWeaver, a handy workflow management system, designed to help us create, manage, and execute the workflow. The foundation workflow skeleton is already created, and all the team members can build on top of it to save huge amount of time avoiding going back and forth to debug technical issues. Most steps are ready to plug and play usage. Geoweaver allows everyone's work to be automatically recorded, shared, utilized, and deployed seamlessly and effortlessly. 

<img width="1205" alt="image" src="https://github.com/geo-smart/swe_forecasting_prod/assets/17322987/899b9694-6863-45d4-9b76-9c90820ced05">

All the source code and history can be retrieved [here](https://github.com/geo-smart/SnowCast)

## Machine Learning Models

The current model is optimized and picked using PyCaret, which is `ExtraTreeRegressor`. The training accuracy is very high (>99%), but it constantly underestimates the high SWE regions, which is understandable, because predicting low values is less penalized in the current training data. 

The other model candidates we want/have tried:
- LSTM
- GRU
- Self Attention
- DNN + DRL (Deep Reinformance Learning)

## Key Problems To Solve

1. Underestimation (overfitting)

The model is seriously overfitting to lean to the low SWE values, and underestimating the mountain alpine snow region right now. We need to make it right by

(1) adjust the sample ratio in the training.csv
(2) adjust the model preference using weights parameters
(3) use reinformace learning to penalize the underestimation and pull it back to be more sensitive to alpine snow area

2. Sensitivity

Right now, we are not very sure how the input data is determining the model behavior or the final results. We need to be more aware of the role each variable is playing. This can be used using feature importance skill like calculating the correlation between input variables and the SNOTEL SWE values. But that is only indirect evaluation. We are brainstorming if there are better ways to do sensitivity analysis.

3. Model Interpretability

Improving our ML models presents a significant challenge primarily due to the inherent ambiguity within the model. We often struggle to discern why the model delivers accurate results on certain occasions while faltering on others. This uncertainty stems from the opaque nature of the model's underlying weights and parameters, obscuring the rationale behind its decisions. To enhance our models, our first imperative is to unravel this ambiguity and gain a comprehensive understanding of their inner workings before embarking on the journey of improvement.Insights gained from model explanations can be used to enhance the model. For example, if a feature is identified as highly influential but its behavior is inconsistent with expectations, it may indicate a need for data preprocessing or feature engineering.

We need a step in the workflow to dedicate for explaining the model for us using languages that human can understand.

4. Model evaluation (during training and after training)

A good model evaluation metric is the key to progress towards the better state in next iteration, instead of going in circles. A model evaluation metric quantifies the model's performance, providing a clear benchmark to gauge its effectiveness. Without a dedicated step for model evaluation, we may lack the means to measure whether our model is getting better or not. Also, by regularly assessing the model against a well-defined evaluation metric, we obtain actionable insights into areas where the model falls short. This enables us to pinpoint weaknesses and make targeted adjustments in the next iteration, guiding the improvement process in a constructive direction. Meanwhile, without continuous evaluation, we risk repeating the same mistakes and encountering similar obstacles in each iteration. An evaluation step acts as a corrective mechanism, steering us away from unproductive cycles and propelling us toward genuine enhancements. Evaluation metrics are data-driven and objective, reducing subjectivity in the assessment process. They provide a solid foundation for making informed decisions about model adjustments and optimizations.

The workflow need a dedicated evaluation step to output the metrics for every run of training and prediction. We also need to determine which evaluation metrics is the best for the model during the training process (loss function, validation metric, optimizers, etc). 

## Tasks to Work On During Hackweek

1. Examine the preprocessing workflow for preparing the training data is correct. 

2. 


## Expected Results

A scalable, reliable, reusable, operational workflow that can be run any time when people want to generate a forecasting SWE map (as long as they have the inputs ready, right now it is meteology and AMSR observation) for western U.S.

The workflow will be one-click exported from the operational Geoweaver instance in the prod EC2 instance, and published in the Github repository. A short paper and documentation will be created to explain the workflow and guide users. 

## How to contribute?

We welcome contributions from the community to help us address the key problems and achieve our goals in developing a robust SWE forecasting workflow for the western U.S. Your contributions can make a significant impact on improving our model and workflow. Here's how you can get involved:

- Code Contributions:

If you have expertise in machine learning, geospatial analysis, or related fields, you can contribute code to enhance our forecasting model and workflow. This may include implementing new algorithms, improving data processing steps, or optimizing existing code.
Fork our GitHub repository and submit pull requests with your code changes. Our team will review and merge the contributions.

- Data Expertise:

Data quality is crucial for accurate forecasting. If you have domain knowledge in meteorology, remote sensing, or related areas, you can help improve our data preprocessing and feature engineering processes.
Share insights and recommendations for better data handling and integration.

- Model Interpretability:

If you specialize in model interpretability techniques, you can help us understand why our model behaves the way it does. Provide guidance on interpreting model results and suggest improvements to enhance interpretability.

- Sensitivity Analysis:

Collaborate with us to explore better ways of conducting sensitivity analysis. Share your expertise in feature importance analysis, correlation evaluation, or other methods to understand how input variables impact model behavior.

- Evaluation Metrics:

Contribute to the selection of appropriate evaluation metrics during model training and post-training evaluation. Help us define the best criteria for assessing model performance, including loss functions, validation metrics, and optimizers.

- Documentation:

Documentation is key to ensuring that our workflow is usable and understandable. Contribute by improving our documentation, writing guides for users, and creating clear explanations of model processes and outcomes.

- Testing and Validation:

Help us test and validate the workflow by running experiments and providing feedback. Identify areas where the model or workflow may need adjustments or fine-tuning.

- Reporting Issues:

If you come across any issues, bugs, or areas for improvement in our workflow, please report them on our GitHub repository's issue tracker. Be sure to provide detailed information about the problem you've encountered.

- Spread the Word:

Share information about our project with your network and encourage others who might be interested in SWE forecasting, machine learning, or geospatial analysis to contribute or use our workflow.

- Feedback:

We value feedback from users and contributors. If you've used our workflow or have ideas for improvements, please share your feedback with us.

## Some Preferred Skills for Quick Start

- Python Programming, better with experiences on Scikit-learn, Pytorch, and Tensorflow or Keras.

- Git/Github branch, merge, commit, PR.

- Linux Shell Scripting.

- Geospatial data processing, like xarray, rasterio, dask, plotly, HDF5, netcdf, zarr, etc.


