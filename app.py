# IMPORTS:
#####################################################################################################
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from app.app_utils.filter_utils import filter_dataframe 


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

    def get_user_selections(self):
        """Get user selections from the sidebar."""
        with st.expander(label='Car selection:', expanded=True):
            left_col, middle_col, right_col = st.columns(spec=[0.35, 0.40, 0.25], gap='large')

            with left_col:
                brands_choice = st.radio('Car Brands List:', ['Common Car Brands', 'All Car Brands'], horizontal=True)

                # default_brand = 'peugeot'
                brands_list = self.performance_df['brand'].str.upper().unique().tolist()
                # default_brand_index = brands_list.index(default_brand)
                selected_brand = st.selectbox(label= f"{brands_choice}:",
                                            options=self.common_brands if brands_choice == 'Common Car Brands' else brands_list,
                                            index= None, # default_brand_index,
                                            placeholder="Select a car brand")
                if selected_brand:
                    selected_brand = selected_brand.lower()
                
                
                brand_models_list = self.performance_df[self.performance_df['brand'] == selected_brand]['model'].unique().tolist()
                # default_model = '3008' if selected_brand == default_brand else None
                # default_model_index = brand_models_list.index(default_model) if default_model else 0
                selected_model = st.selectbox(label='Select model',
                                            options=brand_models_list,
                                            index= None, #default_model_index,
                                            placeholder='Select a model')
                
                if selected_model:
                    selected_transmission = st.multiselect(label='Select Transmission(s)',
                                                        options=['manual', 'automatic'],
                                                        default=['manual', 'automatic'])

                    selected_transmission = [0, 1] if len(selected_transmission) == 2 else ([1] if 'automatic' in selected_transmission else [0])
       
            with middle_col:
                if selected_model:
                    fuel_options = self.results_df[(self.results_df['brand'] == selected_brand) 
                                                        & (self.results_df['model'] == selected_model)]['fuel'].unique()

                    selected_fuel = st.multiselect(label="Select fuel type",
                                                    options=fuel_options,
                                                    default=fuel_options)                  
                    
                    

                    max_km = self.results_df[(self.results_df['brand'] == selected_brand) 
                                                        & (self.results_df['model'] == selected_model)]['km'].dropna().astype(int).max()

                    left_col_km, right_col_km = st.columns(spec=[0.5, 0.5], gap='small')
                    if max_km > 10_000:
                        km_options = np.arange(0, (max_km + 10_000), 10_000)
                        with left_col_km:
                            selected_km_min = st.selectbox(label='Min Km',
                                                    options=km_options,
                                                    index= 0)
                        with right_col_km:
                            selected_km_max = st.selectbox(label='Max Km',
                                                    options=km_options,
                                                    index= (len(km_options)-1)) if len(km_options) > 0 else 0 #default_model_index,
                                                    # placeholder='Select a model')



                    max_age = self.results_df[(self.results_df['brand'] == selected_brand) 
                                                        & (self.results_df['model'] == selected_model)]['age_years'].dropna().astype(int).max()
                    
                    if max_age > 0:
                        age_options = np.arange(0, (max_age + 1), 1)
                        left_col_age, right_col_age = st.columns(spec=[0.5, 0.5], gap='small')
                        with left_col_age:
                            selected_age_min = st.selectbox(label='Min Years',
                                                    options=age_options,
                                                    index= 0)
                        with right_col_age:
                            selected_age_max = st.selectbox(label='Max Years',
                                                    options=age_options,
                                                    index= (len(age_options)-1)) if len(age_options) > 0 else 0 #default_model_index,
                                                    # placeholder='Select a model')
                    else:
                        selected_age_min = 0
                        selected_age_max = 50



                    cv_options = self.results_df[(self.results_df['brand'] == selected_brand) 
                                                        & (self.results_df['model'] == selected_model)]['cv'].dropna().astype(int).sort_values().unique()

                    if cv_options.shape[0] >= 2:
                        left_col_cv, right_col_cv = st.columns(spec=[0.5, 0.5], gap='small')
                        with left_col_cv:
                            selected_cv_min = st.selectbox(label='Select Min Horsepower (CV)',
                                                    options=cv_options,
                                                    index= 0)
                        with right_col_cv:
                            selected_cv_max = st.selectbox(label='Select Max Horsepower (CV)',
                                                    options=cv_options,
                                                    index= (len(cv_options)-1)) if len(cv_options) > 0 else 0 #default_model_index,
                                                    # placeholder='Select a model')
                    else:
                        selected_cv_min = 0
                        selected_cv_max = 1000
                    
                else:
                    selected_transmission = None
                    selected_fuel = None
                    selected_km_min = 0
                    selected_km_max = 500_000
                    selected_age_min = 0
                    selected_age_max = 50
                    selected_cv_min = 0
                    selected_cv_max = 1000

            with right_col:
                if selected_model:
                    # Display car picture
                    car_pictures_brand_model = self.car_pictures_df[
                        (self.car_pictures_df['author'] != 'Unknown') &
                        (self.car_pictures_df['brand'] == selected_brand) &
                        (self.car_pictures_df['model'] == selected_model)
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
                
            
        return selected_brand, selected_model, selected_transmission, \
                selected_fuel, selected_km_min, selected_km_max, selected_age_min, \
                selected_age_max, selected_cv_min, selected_cv_max


    def plot_charts(self, selected_car, selected_brand, selected_model):
        print(selected_car[['cv']])
        """Plot price vs mileage and price vs age in two columns."""
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
        st.title('Car Price Prediction App')

        selected_brand, selected_model, selected_transmission, \
                selected_fuel, selected_km_min, selected_km_max, selected_age_min, \
                selected_age_max, selected_cv_min, selected_cv_max = self.get_user_selections()

        if selected_brand and selected_model:
            selected_car = self.results_df[
                (self.results_df['brand'] == selected_brand) &
                (self.results_df['model'] == selected_model) &
                (self.results_df['is_automatic'].isin(selected_transmission)) &
                (self.results_df['fuel'].isin(selected_fuel)) &
                (self.results_df['km'].between(selected_km_min, selected_km_max)) &
                (self.results_df['age_years'].between(selected_age_min, selected_age_max)) &
                ((self.results_df['cv'].fillna(selected_cv_min).between(selected_cv_min, selected_cv_max)) )
            ]

            self.plot_charts(selected_car, selected_brand, selected_model)

            # st.write("Prediction Results (with all columns):")
            with st.expander(label=f'Our predictions for {selected_brand} {selected_model} are based on this data:', expanded=False):
                edited_df = st.dataframe(filter_dataframe(selected_car[['price', 'predicted_price', 'price_diff', 'km', 'year', 'age_years', 'is_automatic',
                                                       'cv','fuel', 'title']]),
                                        hide_index=True)

# MAIN APP:
#####################################################################################################
if __name__ == "__main__":
    # Initialize the app with paths to your datasets
    app = CarPricePredictionApp(
        performance_data_path='app/app_files/performance_metrics.csv',
        results_data_path='app/app_files/final_results_df.csv',
        pictures_data_path='app/app_files/car_pictures_table_all.csv'
    )
    app.run_app()



#############################################################################################################
