# Tensorcar: your car adviser

## Goal
You want to buy a car. Which model? What price? No worries, Tensorcar will help you get to the right model and right price in no time!

## Motivation
Many of us will need to buy a car at some point. This probably will be the 2nd most expensive thing we buy in our lives.  

Most people are not car-maniacs, so learning the market (product and price) can take considerable time, we can get confused (analysis paralysis), and companies can bias our decision-making.

Tensorcar can help you in taking the best purchasing decision for you. You'll able to see and analyse more information in an unbiased manner, understand what's the right price point, and see how your investment/purchase depreciates over time.
![](https://github.com/sam-math/tensorcar/blob/main/app/app_demo_1.gif)

## Project Scope

- **Current:** At this early stage the scope of the project is limited to Spain (Market) and Price Prediction (App Features).
- **Future:** I'd like to add other European countries, and expand the app to Car Advisory (for example: recommender systems).

## Project structure
This is a full-cycle data science project:

1. We scrape the main spanish car marketplaces monthly (around 200,000 active ads).
2. Preprocess scraped data into a unique tabular dataframe.
3. Understand features distribution to treat outliers, missing data, errors, etc.
4. Augment data and feature-engineer drivers.
5. Use machine-learning to model car prices for each car available (over a thousand different car models).
6. Serve a Streamlit app that gives users the ability to interactively understand car prices with just a few clicks.


## Tech used and Project details
Mostly the project is done in Python

1. **Scraping:** it's done using a combination of requests, playwright, and a thorough analysis of hidden API's. The parameters used and some files aren't available in this GitHub repo to be respectful with the websites from which we scrape.
2. **Data Wrangling:** it's done with Polars, as it's faster than Pandas for the size of the data we are managing (Polars Lazyframes), and its Expressive API allow faster iteration and complex data munging.
3. **EDA and Preprocessing:** it's done with Numpy, Polars, Pandas and Seaborn. Nothing very sophisticated, just knowing how to apply your maths and stats properly (data distribution, long tails, IQRs, CDFs, etc.)
   - Outliers-detection is done at a second stage with regression to ensure outliers "guardrails" account for different price of a car model considering its mileage, age, etc.
5. **Modelling:**
   - A specific model is trainned and saved for each specific car model available (around 1,000 car models).
   - The algorithm/predictor selected for each car depends on the number of car-ads samples available (e.g. Volkswagen Golf has over 5,000 car ads available, Ferrari's just a few dozen or hundred).
   - The algorithms with the hyperparameters chosen are the ones that we found out perform best without going into crazy training times.
   - The overall performance of our predictions is about 93-94% accuracy with no overfitting (absolute price difference between listed price and predicted price for all car ads samples).
     - However, as the objective is not to predict the price of an ad but the "fair" price of a car, we have intentionally decided to tuned our hyperparameters to model for this. Therefore, the overall price variation is 15% (so accuracy of 85%). The key idea is that we want to spot price arbitrage (good and bad ad deals) of future car ads for our users.
6. **User App (Streamlit):**
   - The app provides a car model selector, and allows users adjust key parameters such us mileage, fuel type, gearbox type, etc.
   - Based on this, an interactive scatterplot of price vs mileage and price vs age is displayed, and a polynomic regressor is dynamically trained and display for line of best fit. This helps understand how depreciation works (for example, most premium cars depreciate faster).
   - As well, a price predictor provides a price point estimation for the "average" car of that car model as default. The user can then adjust the parameters to get a price estimation and price range for a car he/she is thinking to buy sell.
   - We have considered user experience for both mobile and desktop users when designing the app (taking into account Streamlit technical capabilities and limitations).
   - In general, early feedback from friends and family has been very positive in terms of predictions and usability. We frequently check that predictions correspond with real car ads, and discrepancies spot possible good deals or expensive ones.

## Feedback and Contact:
If you have any feedback, ideas, etc. more than happy to connect through GitHub or LinkedIn!

Thank you for your time and hope this tool helps you 😊

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License - see the [LICENSE](LICENSE) file for details.
