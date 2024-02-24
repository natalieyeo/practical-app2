We are tasked with predicting used car prices given a dataset with various car features (price, miles on the odometer, make, model, fuel type, transmission, VIN, number of cylinders, size, title status, and condition). After fine-tuning the model to achieve the lowest RMSE possible, I determine the key drivers of used car prices and distill those findings for a non-technical audience of used car dealers.

Step 1: Data understanding and exploration
The original dataset had 400k+ rows including many missing values for all categorical variables, although numerical variables had no missing values. Prices and odometer values had some unbelievable values, indicating a need for major data clean up. In addtion, VINs had duplicated entries, indicating that multiple listings with the same price existed for the same car in different regions. As a result, I decided not to include region in the analysis.

Step 2: Data preparation
I removed significant outliers in the data, as well rows with duplicated VIN numbers, changing the VIN variable to "known"/"unknown" and filling missing values of all other
categorical columns with "unknown." I also consolidated the 20k+ unique car models into 925 models so that we could better predict how a car's make and model affect the price.

Step 3: Modelling
In this section, I experimented with using multiple subsections of the dataset to build various models.

Dataset 1 includes all columns in the cleaned dataset from the previous section. This dataset had 210k+ rows.

In Dataset 2, I hypothesized that the manufacturer, model, odometer, year, and condition of the car had the highest impact on the price car, so I removed all other features. I also cleaned the data further by removing the rows where the condition of the car was unknown and removing models of cars that were unknown or 'other' or did not fall into the top 50 models as determined by the value counts. This resulted in roughly 65k rows.

In Dataset 3, I included entries where the condition of the car was unknown, but excluded entries where the manufacturer and models were unknown or other. I theorized that condition of the car would not impact the metrics so much and removed the column. I also excluded all other features except manufacturer, model, odometer, and year. I concatenated the manufacterer and model columns to create a 'type' column so that the make and model were included in one variable. After that, I dropped the individual manufacturer and model columns. This resulted in a dataset with about 190k rows.

Models:
Feature selection using LASSO paired with Linear Regression and various degrees of Polynomial Features
Using LASSO model only without feature selection, grid searching on best alphas and tolerance
Feature selection using SequentialFeatureSelector() and Linear Regression
Feature selection using Recursive Feature Elimination and Linear Regression

Categorical encoding experiments:
Using OneHotEncoder() vs James Stein Encoder() for categorical encoding
I theorized that it would be easier for a model to predict prices of cars that had more common makes and models, so I limited the dataset to the 50 most common car models and created 49 different car model variables using OneHotEncoder. This was useful in determining which car models had the highest effect on price.

Key Takeaways:
Overall, all the modelling we have done reinforces the idea that the most important features in a used car are the number of miles on the odometer, year of the car, and the car's make and model. Permutation feature further supports this, although it varies based on the model. With a model where the dataset includes unknown car models and unknown condition of the car, there is more noise, and "odometer" is by far the most important variable, followed by "model" and "year." However, when we vet the dataset more thoroughly to only known models and known condition of the car, the concated model and manufacturer variable "type" is by far the most important variable, followed by year. Removing the "type" of car would decrease the accuracy score of said model by almost 60%, and removing "year" would decrease the accuracy score by slightly over 50%. Compared to the first model, odometer has much less of an impact.

The best RMSE was 7304, which means that we were able to predict the price within about 7000 of the actual price. This RMSE was achieved on dataset 2, a heavily scrubbed dataset including only the top 50 models of cars, condition, year, and odometer. LASSO feature selection with Linear Regression was consistently the better model compared to other feature selection techniques. And using the OneHotEncoder to differentiate between the relationship of car model to price lends insightful data which we will delve into in the the deployment section.

A simpler model with less noise performed better than models including all the features (such as fuel type, cylinders, whether the VIN number is known, and transmission). The One Hot Encoder seemed to result in more accurate predictions than the James Stein Encoder, but James Stein was still useful when the number of unique categories was in the hundreds.

The relationship between price, year, and miles on the odometer is not exactly linear. The model has smaller root mean squared error for both training and test sets using higher degree polynomials of 2 and 3. Though mean squared error is lower for degree 3 polynomial features, the difference in RMSE between degree 2 and 3 is not much.

The main takeaway for car dealers is not a new one: odometer, year, make, and model are the most influential drives of car prices. Models that seem to hold the most value tend to be large trucks. Dodge RAM 3500 and 2500, Chevrolet 3500 and 2500, and Ford F-350s consistently tend to have the largest coefficients, representing the highest prices. Also high on the list of most impactful coefficients are Corvettes and Mercedes Benzes. Budget vehicles such as the Chevrolet Cruze, Nissan Sentra, Ford Fusion, Nissan Altima, and Hyundai Elantra had the most negative impact on car prices.

It is important to note that since the sample size of high-end luxury vehicles like Ferraris and Porsches was so so low, the model could not ascertain the impact of these brands on the vehicle price.
