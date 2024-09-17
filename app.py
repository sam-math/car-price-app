# IMPORTS:
#####################################################################################################
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


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

    def get_user_selections(self):
        """Get user selections from the sidebar."""
        with st.sidebar:
            default_brand = 'peugeot'
            brands_list = self.performance_df['brand'].unique().tolist()
            default_brand_index = brands_list.index(default_brand)
            selected_brand = st.selectbox(label='Select Brand',
                                          options=brands_list,
                                          index=default_brand_index,
                                          placeholder="Select a car brand")

            brand_models_list = self.performance_df[self.performance_df['brand'] == selected_brand]['model'].unique().tolist()
            default_model = '3008' if selected_brand == default_brand else None
            default_model_index = brand_models_list.index(default_model) if default_model else 0
            selected_model = st.selectbox(label='Select model',
                                          options=brand_models_list,
                                          index=default_model_index,
                                          placeholder='Select a model')

            selected_transmission = st.multiselect(label='Select Transmission(s)',
                                                   options=['manual', 'automatic'],
                                                   default=['manual', 'automatic'])

            selected_transmission = [0, 1] if len(selected_transmission) == 2 else ([1] if 'automatic' in selected_transmission else [0])

            selected_km = st.slider(label='Select mileage (km)',
                                    min_value=0,
                                    max_value=500_000,
                                    value=(0, 200_000),
                                    step=25_000)

            selected_age = st.slider(label='Select age of car (years)',
                                     min_value=0,
                                     max_value=25,
                                     value=(0, 12),
                                     step=1)

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
                st.image(image_picture_url, use_column_width=True)
                st.info(f"Image by: [{image_author}]({image_full_url}), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) via Wikimedia Commons")
            else:
                st.info('Car image not available')

        return selected_brand, selected_model, selected_transmission, selected_km, selected_age

    def plot_charts(self, selected_car, selected_brand, selected_model):
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

                st.plotly_chart(fig, use_container_width=True)

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

                st.plotly_chart(fig_age, use_container_width=True)

            else:
                st.warning("No data available for this brand-model combination.")

    def run_app(self):
        """Main entry point to run the Streamlit app."""
        st.title('Car Price Prediction App')

        selected_brand, selected_model, selected_transmission, selected_km, selected_age = self.get_user_selections()

        if selected_brand and selected_model:
            selected_car = self.results_df[
                (self.results_df['brand'] == selected_brand) &
                (self.results_df['model'] == selected_model) &
                (self.results_df['is_automatic'].isin(selected_transmission)) &
                (self.results_df['km'].between(selected_km[0], selected_km[1])) &
                (self.results_df['age_years'].between(selected_age[0], selected_age[1]))
            ]

            self.plot_charts(selected_car, selected_brand, selected_model)

            st.write("Prediction Results (with all columns):")
            edited_df = st.data_editor(selected_car, use_container_width=True)

# MAIN APP:
#####################################################################################################
if __name__ == "__main__":
    # Initialize the app with paths to your datasets
    app = CarPricePredictionApp(
        performance_data_path='06_app/app_files/performance_metrics.csv',
        results_data_path='06_app/app_files/final_results_df.csv',
        pictures_data_path='06_app/app_files/car_pictures_table_all.csv'
    )
    app.run_app()
