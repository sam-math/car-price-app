# IMPORTS:
#####################################################################################################
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from app.app_utils.filter_utils import filter_dataframe 
import joblib


# CONFIG OPTIONS:
#####################################################################################################
st.set_page_config(layout="wide")


# FUNCTION DEFINITIONS:
#####################################################################################################

class CarPricePredictionApp:
    
    def __init__(self, performance_data_path, results_data_path, pictures_data_path):
        self.performance_df = pd.read_csv(performance_data_path)
        self.results_df = pd.read_csv(results_data_path)
        self.car_pictures_df = pd.read_csv(pictures_data_path)
        self.common_brands = ['ALFA ROMEO', 'AUDI', 'BMW', 'CITROEN', 'CUPRA', 'DACIA', 'DS', 'FIAT', 'FORD',
                            'HONDA', 'HYUNDAI', 'JEEP', 'KIA', 'LAND-ROVER', 'LEXUS', 'MAZDA', 'MERCEDES-BENZ',
                            'MG', 'MINI', 'MITSUBISHI', 'NISSAN', 'OPEL', 'PEUGEOT', 'RENAULT', 'ROVER', 'SEAT',
                            'SKODA', 'SUZUKI', 'TESLA', 'TOYOTA', 'VOLKSWAGEN', 'VOLVO']
        # User selections will be stored as instance variables
        self.selected_brand = None
        self.selected_model = None
        self.selected_transmission = None
        self.fuel_options = None
        self.selected_fuel = None
        self.selected_km_min = None
        self.selected_km_max = None
        self.selected_age_min = None
        self.selected_age_max = None
        self.cv_options = None
        self.selected_cv_min = None
        self.selected_cv_max = None
        self.selected_car = None
        self.price_prediction = None
        self.selected_car_metrics = None
        self.low_margin_value = None
        self.low_bound_value = None
        self.mid_low_bound_value = None
        self.mid_high_bound_value = None
        self.high_bound_value = None
        self.high_margin_value = None

    def get_user_selections(self):
        """Get user selections from the sidebar and store them as attributes."""
        with st.expander(label='Car selection:', expanded=True):
            left_col, middle_col, right_col = st.columns(spec=[0.35, 0.40, 0.25], gap='large')

            with left_col:
                brands_choice = st.radio('Car Brands List:', ['Popular Car Brands', 'All Car Brands'], horizontal=True)

                brands_list = self.performance_df['brand'].str.upper().unique().tolist()
                self.selected_brand = st.selectbox(
                    label=f"{brands_choice}:",
                    index=None,
                    options=self.common_brands if brands_choice == 'Popular Car Brands' else brands_list,
                    placeholder="Select a car brand"
                )
                if self.selected_brand:
                    self.selected_brand = self.selected_brand.lower()

                brand_models_list = self.performance_df[self.performance_df['brand'] == self.selected_brand]['model'].unique().tolist()
                self.selected_model = st.selectbox(
                    label='Select model',
                    index=None,
                    options=brand_models_list,
                    placeholder='Select a model'
                )

                if self.selected_model:
                    self.selected_transmission = st.multiselect(
                        label='Select Transmission(s)',
                        options=['manual', 'automatic'],
                        default=['manual', 'automatic']
                    )
                    self.selected_transmission = [0, 1] if len(self.selected_transmission) == 2 else (
                        [1] if 'automatic' in self.selected_transmission else [0])

            with middle_col:
                if self.selected_model:
                    self.fuel_options = self.results_df[(self.results_df['brand'] == self.selected_brand)
                                                   & (self.results_df['model'] == self.selected_model)]['fuel'].unique()
                    self.selected_fuel = st.multiselect(label="Select fuel type", options=self.fuel_options, default=self.fuel_options)

                    max_km = self.results_df[(self.results_df['brand'] == self.selected_brand)
                                             & (self.results_df['model'] == self.selected_model)]['km'].dropna().astype(int).max()
                    if max_km > 10_000:
                        km_options = np.arange(0, (max_km + 10_000), 10_000)
                        left_col_km, right_col_km = st.columns(spec=[0.5, 0.5], gap='small')
                        with left_col_km:
                            self.selected_km_min = st.selectbox(label='Min Km', options=km_options, index=0, format_func=lambda x: f"{x:,}")
                        with right_col_km:
                            self.selected_km_max = st.selectbox(label='Max Km', options=km_options, index=(len(km_options) - 1), format_func=lambda x: f"{x:,}")

                    max_age = self.results_df[(self.results_df['brand'] == self.selected_brand)
                                              & (self.results_df['model'] == self.selected_model)]['age_years'].dropna().astype(int).max()
                    if max_age > 0:
                        age_options = np.arange(0, (max_age + 1), 1)
                        left_col_age, right_col_age = st.columns(spec=[0.5, 0.5], gap='small')
                        with left_col_age:
                            self.selected_age_min = st.selectbox(label='Min Years', options=age_options, index=0)
                        with right_col_age:
                            self.selected_age_max = st.selectbox(label='Max Years', options=age_options, index=(len(age_options) - 1))
                    else:
                        self.selected_age_min, self.selected_age_max = 0, 50

                    self.cv_options = self.results_df[(self.results_df['brand'] == self.selected_brand)
                                                 & (self.results_df['model'] == self.selected_model)]['cv'].dropna().astype(int).sort_values().unique()
                    if self.cv_options.shape[0] >= 2:
                        left_col_cv, right_col_cv = st.columns(spec=[0.5, 0.5], gap='small')
                        with left_col_cv:
                            self.selected_cv_min = st.selectbox(label='Select Min Horsepower (CV)', options=self.cv_options, index=0)
                        with right_col_cv:
                            self.selected_cv_max = st.selectbox(label='Select Max Horsepower (CV)', options=self.cv_options, index=(len(self.cv_options) - 1))
                    else:
                        self.selected_cv_min, self.selected_cv_max = 0, 1000
            
            with right_col:
                if self.selected_model:
                    # Display car picture
                    car_pictures_brand_model = self.car_pictures_df[
                        (self.car_pictures_df['author'] != 'Unknown') &
                        (self.car_pictures_df['brand'] == self.selected_brand) &
                        (self.car_pictures_df['model'] == self.selected_model)
                    ].reset_index(drop=True)

                    if not car_pictures_brand_model.empty:
                        selected_picture = car_pictures_brand_model.iloc[0]
                        image_author = selected_picture['author']
                        image_full_url = selected_picture['fullurl']
                        image_picture_url = selected_picture['imageinfo.0.thumburl']
                        image_attribution = f"""Image by: [{image_author}]({image_full_url}),
                                                [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0)"""
                                                # via Wikimedia Commons"""
                        st.image(image_picture_url, use_column_width='auto')
                        
                        st.info(image_attribution, icon=":material/attribution:")
                    else:
                        st.info('Car image not available')

    def filter_car_data(self):
        """Filter car data based on the user selections."""
        return self.results_df[
            (self.results_df['brand'] == self.selected_brand) &
            (self.results_df['model'] == self.selected_model) &
            (self.results_df['is_automatic'].isin(self.selected_transmission)) &
            (self.results_df['fuel'].isin(self.selected_fuel)) &
            (self.results_df['km'].between(self.selected_km_min, self.selected_km_max)) &
            (self.results_df['age_years'].between(self.selected_age_min, self.selected_age_max)) &
            (self.results_df['cv'].fillna(self.selected_cv_min).between(self.selected_cv_min, self.selected_cv_max))
        ]
    
    def plot_price_range(self):
        # Define price categories, ranges, and corresponding colors
        categories = ['Great', 'Good', 'Fair', 'Tight', 'Expensive']
        price_ranges = [(self.low_margin_value, self.low_bound_value),
                        (self.low_bound_value, self.mid_low_bound_value),
                        (self.mid_low_bound_value, self.mid_high_bound_value),
                        (self.mid_high_bound_value, self.high_bound_value),
                        (self.high_bound_value, self.high_margin_value)]
        colors = ['#368A65', '#40B36A', '#BCD980', '#FAFFA1', '#F4CD96']  # Colors for the stacked bars

        # Create a vertical stacked bar chart
        fig, ax = plt.subplots(figsize=(4, 4))  # Adjust size for vertical layout

        # Start stacking the bars
        bottom_position = 0
        for i, price_range in enumerate( price_ranges):
            # Calculate the height of the current category
            height = price_range[1] - price_range[0]
            ax.bar([0], height, bottom=bottom_position, color=colors[i], edgecolor='white')
            bottom_position += height

        # Add the price ranges as text on the bars
        bottom_position = 0
        for i, price_range in enumerate(price_ranges):
            mid_position = bottom_position + (price_range[1] - price_range[0]) / 2
            ax.text(0, mid_position, f"{categories[i]}: {price_range[0]:,}€ - {price_range[1]:,}€", 
                    ha='center', va='center', fontsize=10, color='black')
            bottom_position += price_range[1] - price_range[0]
            ax.axis('off')
        plt.box(False)

        # Display the plot in Streamlit
        st.pyplot(fig, use_container_width=True)

    def predict_price(self):        
        if self.selected_model:
            with st.expander("Price Prediction", expanded=True):
                
                col_1, col_2 = st.columns([0.7, 0.3], gap='large')
                # col_1, col_2, col_3, col_4, col_5 = st.columns([0.23, 0.23, 0.23, 0.23, 0.08])
                with col_1:
                    user_km = st.number_input('Km:', value=round(self.selected_car['km'].median().astype(int),-4), step=5000)
                # with col_2:
                    user_age_car = st.number_input('Car Age:', value=self.selected_car['age_years'].median().astype(int))
                # with col_3:
                    idx_fuel = self.fuel_options.tolist().index(self.selected_car['fuel'].mode().values[0])
                    user_fuel = st.selectbox("Fuel:", options= self.fuel_options, index=idx_fuel)
                # with col_4:
                    try:
                        idx_cv = self.cv_options.tolist().index(self.selected_car['cv'].mode().values[0])
                        user_cv = st.selectbox("Horsepower (CV):", options=self.cv_options, index=idx_cv)
                    except:
                        user_cv = None
                # with col_5:
                    st.text('')
                    user_automatic = st.checkbox('Is Automatic', value=self.selected_car['is_automatic'].mode().values[0])
                

                
                
                    user_input = {
                        'km': user_km,  # e.g., 50000
                        'age_years': user_age_car,  # e.g., 3
                        'is_automatic': user_automatic,  # 1 if automatic, 0 if manual
                        'fuel': user_fuel,  # e.g., 'Diesel'
                        'cv': user_cv  # e.g., '150'
                    }

                    # Load the model and feature names
                    try:
                        model_info = joblib.load(f'05_model/saved_models/{self.selected_brand}_{self.selected_model}.pkl')
                    except Exception as e:
                        st.error(f"Error type: {e}")
                    model = model_info['model']
                    feature_names = model_info['feature_names']

                    # Prepare user input for prediction (reindex with feature_names)
                    user_df = pd.DataFrame([user_input])
                    user_df = pd.get_dummies(user_df, columns=['fuel', 'cv'])
                    user_df = user_df.reindex(columns=feature_names, fill_value=0)

                    # Make prediction
                    self.price_prediction = model.predict(user_df)[0]
                    # _, mid, _ = st.columns([0.35, 0.15, 0.40])
                    # with mid:

                    sub_col1, sub_col2 = st.columns(2)
                    with sub_col1:
                        st.success(f"**Predicted price: {int(self.price_prediction):,}€**")

                    with sub_col2:
                   
                        self.selected_car_metrics = (self.performance_df[(self.performance_df['brand'] == self.selected_brand)
                                                    &
                                                    (self.performance_df['model'] == self.selected_model)])
                        # st.write(self.selected_car_metrics)
                        wide_bound = (self.selected_car_metrics['iqr_10_90']/2).values[0]
                        mid_bound = (self.selected_car_metrics['iqr_25_75']/2).values[0]

                        price_prediction_rounded = int(self.price_prediction)
                        # st.write(model_info['training_samples'])
                        st.info(f"**80% of offers are predicted to be between {int(round(self.price_prediction - wide_bound*2,-2)):,}€ and {int(round(self.price_prediction + wide_bound*2,-2)):,}€**")

                    mid_bound_capped = max(min(600,mid_bound), 200) 
                    wide_bound_capped = max(min(mid_bound_capped, wide_bound), mid_bound_capped + 1000)
                    margin_bound_capped = max(wide_bound, wide_bound_capped + 1000)

                    self.mid_low_bound_value = round((self.price_prediction - mid_bound_capped).astype(int),-2)
                    self.mid_high_bound_value = round((self.price_prediction + mid_bound_capped).astype(int),-2)

                    self.low_margin_value = round((self.price_prediction - margin_bound_capped).astype(int),-2)
                    self.low_bound_value = round((self.price_prediction - wide_bound_capped).astype(int),-2)
                    self.mid_low_bound_value = round((self.price_prediction - mid_bound_capped).astype(int),-2)
                    self.mid_high_bound_value = round((self.price_prediction + mid_bound_capped).astype(int),-2)
                    self.high_bound_value = round((self.price_prediction + wide_bound_capped).astype(int),-2)
                    self.high_margin_value = round((self.price_prediction + margin_bound_capped).astype(int),-2)
                
                with col_2:
                    self.plot_price_range()
                
                




    def plot_charts(self, selected_car, selected_brand, selected_model):
        left_col, right_col = st.columns(2)
        with left_col:
            X = selected_car[['km']]
            y = selected_car['price']

            if X.shape[0] > 0:
                poly = PolynomialFeatures(degree=3)
                X_poly = poly.fit_transform(X)

                model = LinearRegression()
                model.fit(X_poly, y)

                X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                X_range_poly = poly.transform(X_range)
                y_pred = model.predict(X_range_poly)

                fig = px.scatter(selected_car, x='km', y='price', color='is_automatic', opacity=0.5,
                                 labels={'km': 'Mileage (km)', 'price': 'Price'},
                                 title=f'Price vs Mileage (km) for {selected_brand} {selected_model}',
                                 color_discrete_map={1: '#002776', 0: '#92d400'})

                fig.add_scatter(x=X_range.flatten(), y=y_pred, mode='lines', showlegend=False,
                                line=dict(color='#0b7ca3', width=3), opacity=0.7)

                fig.update_layout(legend=dict(x=1, y=1, traceorder='normal', orientation='v',
                                              xanchor='right', yanchor='top'))

                st.plotly_chart(fig, use_container_width=True, config={'staticPlot':True})

            else:
                st.warning("No data available for this brand-model combination.")

        with right_col:
            X_age = selected_car[['age_years']]
            y_age = selected_car['price']

            if X_age.shape[0] > 0:
                poly_age = PolynomialFeatures(degree=3)
                X_age_poly = poly_age.fit_transform(X_age)

                model_age = LinearRegression()
                model_age.fit(X_age_poly, y_age)

                X_age_range = np.linspace(X_age.min(), X_age.max(), 100).reshape(-1, 1)
                X_age_range_poly = poly_age.transform(X_age_range)
                y_age_pred = model_age.predict(X_age_range_poly)

                fig_age = px.scatter(selected_car, x='age_years', y='price', color='is_automatic', opacity=0.3,
                                     labels={'age_years': 'Car Age (years)', 'price': 'Price'},
                                     title=f'Price vs Car Age (years) for {selected_brand} {selected_model}',
                                     color_discrete_map={1: '#002776', 0: '#92d400'})

                fig_age.add_scatter(x=X_age_range.flatten(), y=y_age_pred, mode='lines', showlegend=False,
                                    line=dict(color='#0b7ca3', width=3), opacity=0.8)

                fig_age.update_layout(legend=dict(x=1, y=1, traceorder='normal', orientation='v',
                                                  xanchor='right', yanchor='top'))

                st.plotly_chart(fig_age, use_container_width=True, config={'staticPlot':True})

            else:
                st.warning("No data available for this brand-model combination.")

    def run_app(self):
            """Main entry point to run the Streamlit app."""
        #     st.markdown(
        #     """
        #     <style>
        #     .custom-title {
        #         font-size: 48px; /* Font size */
        #         font-weight: bold; /* Bold font */
        #         text-align: center; /* Center the text */
        #         background: linear-gradient(90deg, #05384d, #066db3, #22c897); /* Gradient colors */
        #         -webkit-background-clip: text; /* Clip the background to the text */
        #         -webkit-text-fill-color: transparent; /* Fill the text with transparent color */
        #         margin-bottom: 20px; /* Space below the title */
        #         padding: 20px; /* Add padding for better spacing */
        #         border-radius: 10px; /* Rounded corners */
        #         font-family: 'Cabin', sans-serif;
        #     }
        #     </style>
        #     <h1 class="custom-title">Tensorcar: Your Car Price Predictor</h1>
        #     """, 
        #     unsafe_allow_html=True
        # )
            st.title('Tensorcar: Your Car Predictor App')

            self.get_user_selections()

            if self.selected_brand and self.selected_model:
                self.selected_car = self.filter_car_data()
                self.plot_charts(self.selected_car, self.selected_brand, self.selected_model)

                self.predict_price()

                with st.expander(label=f'Our predictions for {self.selected_brand} {self.selected_model} are based on this data:', expanded=False):
                    edited_df = st.dataframe(filter_dataframe(self.selected_car[['price', 'predicted_price', 'price_diff', 'km', 'year', 'age_years', 'is_automatic',
                                                        'cv','fuel', 'title']]),
                                            hide_index=True)

                    

# MAIN APP:
#####################################################################################################
if __name__ == "__main__":
    # Initialize the app with paths to your datasets
    # st.markdown(
    # """
    # <style>
    # .main {
    #     background: linear-gradient(120deg, rgba(241,247,247,1) 48%, rgba(179,197,223,1) 100%);
    #     padding: 20px;
    #     border-radius: 10px;
    #     color: black;
    # }
    # .stApp {
    #     background-color: #090979;
    # }
    # </style>
    # """, 
    # unsafe_allow_html=True
    # )

    app = CarPricePredictionApp(
        performance_data_path='app/app_files/performance_metrics.csv',
        results_data_path='app/app_files/final_results_df.csv',
        pictures_data_path='app/app_files/car_pictures_table_all.csv'
    )
    app.run_app()
    


#############################################################################################################
