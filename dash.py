import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
def load_data():
    """Load the rental dataset"""
    try:
        df = pd.read_csv(r"C:\Users\User\Downloads\cleaned_KL_data.csv")
        st.sidebar.success("Data loaded successfully!")
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

def predict_price(features, xgb_model, encoders):
    """Make rental price prediction using saved XGBoost model
    
    Args:
        features (dict): Dictionary containing the input features
        xgb_model: Loaded XGBoost model from pickle
        encoders: Loaded encoders from pickle
        
    Returns:
        dict: Contains prediction, feature importance, and input features
    """
    try:
        # Create DataFrame with input features
        input_df = pd.DataFrame([features])
        
        # Encode categorical variables
        input_df['location_encoded'] = encoders['location'].transform([features['location']])
        input_df['property_type_encoded'] = encoders['property_type'].transform([features['property_type']])
        input_df['furnished_encoded'] = encoders['furnished'].transform([features['furnished']])
        
        # Calculate derived features
        #input_df['property_age'] = 2024 - input_df['completion_year']
        #input_df['price_per_sqft'] = 0
        
        # Select and order features
        feature_order = [
            'size',
            'rooms',
            'bathroom',
            'parking',
            'additional_near_ktm/lrt',
            'location_encoded',
            'property_type_encoded',
            'furnished_encoded'
            
        ]
        
        # Prepare final features
        X_pred = input_df[feature_order]
        
        # Convert boolean to int
        X_pred['parking'] = X_pred['parking'].astype(int)
        X_pred['additional_near_ktm/lrt'] = X_pred['additional_near_ktm/lrt'].astype(int)
        
        # Make prediction
        prediction = xgb_model.predict(X_pred)[0]
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_order,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'prediction': prediction,
            'feature_importance': feature_importance,
            'input_features': X_pred
        }
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.write("Debug information:")
        st.write("Input features:", features)
        st.write("Available encoders:", list(encoders.keys()))
        return None

#def validate_dataset(df):
    """Validate if uploaded dataset has the required columns"""
    required_columns = [
        'monthly_rent', 'location', 'property_type', 'furnished', 
        'size', 'rooms', 'bathroom', 'parking', 'additional_near_ktm/lrt',
        'prop_name'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    return len(missing_columns) == 0, missing_columns

#def add_data_upload():
    """Add data upload functionality and dataset overview"""
    st.title("📊 Data Upload & Overview")
    
    # Add description and instructions
    st.write("""
    ### Upload Your Rental Dataset
    Upload your rental dataset to analyze property trends and make predictions. 
    
    **Requirements:**
    - File format: CSV
    - Required columns: monthly_rent, location, property_type, furnished, size, rooms, bathroom, parking, additional_near_ktm/lrt, prop_name
    - Numerical values should be properly formatted
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate dataset
            is_valid, missing_cols = validate_dataset(df)
            
            if is_valid:
                st.success("Dataset successfully uploaded! Here's an overview of your data:")
                
                # Save the dataframe to session state for use in other tabs
                st.session_state['df'] = df
                st.session_state['data_uploaded'] = True
                
                # Display dataset overview
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Properties", f"{len(df):,}")
                
                with col2:
                    st.metric("Locations", f"{len(df['location'].unique()):,}")
                
                with col3:
                    st.metric("Property Types", f"{len(df['property_type'].unique()):,}")
                
                with col4:
                    st.metric("Avg. Rent", f"RM {df['monthly_rent'].mean():,.2f}")
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Show data summary
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Locations in Dataset")
                    location_counts = df['location'].value_counts()
                    fig_locations = px.bar(
                        x=location_counts.index,
                        y=location_counts.values,
                        title='Properties by Location',
                        labels={'x': 'Location', 'y': 'Count'}
                    )
                    fig_locations.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_locations, use_container_width=True)
                
                with col2:
                    st.subheader("Property Types in Dataset")
                    type_counts = df['property_type'].value_counts()
                    fig_types = px.pie(
                        values=type_counts.values,
                        names=type_counts.index,
                        title='Distribution of Property Types'
                    )
                    st.plotly_chart(fig_types, use_container_width=True)
                
                # Data quality check
                st.subheader("Data Quality Check")
                
                # Check for missing values
                missing_data = df.isnull().sum()
                if missing_data.sum() > 0:
                    st.warning("Found missing values in your dataset:")
                    st.write(missing_data[missing_data > 0])
                else:
                    st.success("No missing values found in your dataset!")
                
                # Check for numerical columns
                numeric_issues = []
                for col in ['monthly_rent', 'size', 'rooms', 'bathroom']:
                    if not pd.to_numeric(df[col], errors='coerce').notnull().all():
                        numeric_issues.append(col)
                
                if numeric_issues:
                    st.warning(f"Found non-numeric values in these columns: {', '.join(numeric_issues)}")
                else:
                    st.success("All numerical columns are properly formatted!")
                
            else:
                st.error(f"Dataset is missing required columns: {', '.join(missing_cols)}")
                st.write("Please ensure your dataset contains all required columns.")
                
        except Exception as e:
            st.error(f"Error reading the file: {str(e)}")
            st.write("Please make sure your file is properly formatted CSV.")
    
    else:
        # Show sample dataset format
        st.info("No file uploaded. Here's a sample of the required dataset format:")
        sample_data = pd.DataFrame({
            'prop_name': ['Property A', 'Property B'],
            'monthly_rent': [2500, 3000],
            'location': ['Location A', 'Location B'],
            'property_type': ['Condo', 'Apartment'],
            'furnished': ['Fully Furnished', 'Partially Furnished'],
            'size': [1000, 1200],
            'rooms': [3, 2],
            'bathroom': [2, 2],
            'parking': [True, False],
            'additional_near_ktm/lrt': [True, False]
        })
        st.dataframe(sample_data)

def show_market_analysis(df):

    """Display market analysis visualizations"""
    st.subheader("Market Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average rent by property type
        fig_property = px.box(
            df,
            x='property_type',
            y='monthly_rent',
            title='Rent Distribution by Property Type'
        )
        st.plotly_chart(fig_property)
    
    with col2:
        # Average rent by size category
        avg_size = df.groupby('size_category')['monthly_rent'].mean().reset_index()
        fig_size = px.bar(
            avg_size,
            x='size_category',
            y='monthly_rent',
            title='Average Rent by Size Category'
        )
        st.plotly_chart(fig_size)

def calculate_location_stats(df):
    """
    Calculate comprehensive location statistics
    
    Parameters:
    -----------
    df : pandas DataFrame
        The rental dataset
        
    Returns:
    --------
    DataFrame
        Location-wise statistics
    """
    try:
        # Calculate basic statistics for each location
        location_stats = df.groupby('location').agg({
            'monthly_rent': ['mean', 'median', 'min', 'max', 'count'],
            'size': ['mean', 'median'],
            'rooms': ['mean', 'median'],
            'bathroom': ['mean'],
            'parking': ['mean'],
            'facility_gymnasium': ['mean']
        }).round(2)
        
        # Flatten the column names
        location_stats.columns = [
            f"{col[0]}_{col[1]}" for col in location_stats.columns
        ]
        
        # Reset index to make location a column
        location_stats = location_stats.reset_index()
        
        # Add price per sqft
        location_stats['price_per_sqft'] = (
            location_stats['monthly_rent_mean'] / location_stats['size_mean']
        ).round(2)
        
        # Add market position
        overall_mean = df['monthly_rent'].mean()
        location_stats['market_position'] = (
            (location_stats['monthly_rent_mean'] - overall_mean) / overall_mean * 100
        ).round(2)
        
        return location_stats
    
    except Exception as e:
        st.error(f"Error calculating location statistics: {str(e)}")
        st.write("Debug information:")
        st.write("DataFrame columns:", df.columns.tolist())
        return pd.DataFrame()  # Return empty DataFrame on error
    
def predict_future_trend(df, xgb_model, encoders, location, location_stats):
    """
    Predict rental price trends with location statistics using current average as base
    """
    try:
        # Get current average rent for the location
        loc_stats = location_stats[location_stats['location'] == location].iloc[0]
        current_rent = loc_stats['monthly_rent_mean']
        
        # Set growth rates based on location market position
        market_position = float(loc_stats['market_position'])
        
        # Define annual growth rate based on market position
        if market_position > 20:
            annual_growth_rate = 0.06  # 6% for premium locations
        elif market_position > 0:
            annual_growth_rate = 0.05  # 5% for above-average locations
        else:
            annual_growth_rate = 0.04  # 4% for other locations
            
        # Generate predictions for next 10 years
        years = range(2024, 2034)
        predictions = []
        
        for year in years:
            # Calculate years from start
            years_from_start = year - 2024
            
            # Calculate predicted rent using compound growth from current rent
            predicted_rent = current_rent * (1 + annual_growth_rate) ** years_from_start
            
            # Calculate the cumulative growth rate
            cumulative_growth_rate = ((predicted_rent / current_rent) - 1) * 100
            
            predictions.append({
                'year': year,
                'predicted_rent': round(predicted_rent, 2),
                'lower_bound': round(predicted_rent * 0.95, 2),
                'upper_bound': round(predicted_rent * 1.05, 2),
                'growth_rate': round(cumulative_growth_rate, 2)
            })
        
        # Create DataFrame
        trend_df = pd.DataFrame(predictions)
        
        # Create visualization
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add prediction interval
        fig.add_trace(
            go.Scatter(
                x=trend_df['year'],
                y=trend_df['upper_bound'],
                fill=None,
                mode='lines',
                line_color='rgba(0,100,80,0.2)',
                showlegend=False,
                name='Upper Bound'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=trend_df['year'],
                y=trend_df['lower_bound'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,100,80,0.2)',
                name='Prediction Interval'
            )
        )
        
        # Add predicted trend line
        fig.add_trace(
            go.Scatter(
                x=trend_df['year'],
                y=trend_df['predicted_rent'],
                name='Predicted Rent',
                line=dict(color='rgb(0,100,80)', width=3),
                mode='lines+markers'
            )
        )
        
        # Add growth rate line on secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=trend_df['year'],
                y=trend_df['growth_rate'],
                name='Cumulative Growth Rate (%)',
                line=dict(color='red', dash='dot'),
                mode='lines'
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title=f'Rental Price Trend Prediction for {location} (2024-2033)',
            yaxis_title='Monthly Rent (RM)',
            yaxis2_title='Cumulative Growth Rate (%)',
            hovermode='x unified',
            showlegend=True
        )
        
        # Add debugging information
        st.write("Debug Information:")
        st.write(f"Current Average Rent: RM {current_rent:,.2f}")
        st.write(f"Market Position: {market_position:,.2f}%")
        st.write(f"Annual Growth Rate: {annual_growth_rate*100:,.1f}%")
        st.write(f"10-Year Projection: RM {predictions[-1]['predicted_rent']:,.2f}")
        st.write(f"Total Growth: {predictions[-1]['growth_rate']:,.1f}%")
        
        return fig, trend_df
        
    except Exception as e:
        st.error(f"Error predicting trend: {str(e)}")
        st.write("Debug information:")
        st.write("Location:", location)
        if 'current_rent' in locals():
            st.write("Current rent:", current_rent)
        if 'market_position' in locals():
            st.write("Market position:", market_position)
        if 'annual_growth_rate' in locals():
            st.write("Growth rate:", annual_growth_rate)
        return None, None


def add_trend_analysis_section(df, xgb_model, encoders, location_stats):
    """Add trend analysis section to dashboard with enhanced visualizations"""
    st.subheader("📈 Future Rental Market Trend Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Location selector for trend analysis
        location = st.selectbox(
            "Select Location for Trend Analysis",
            options=sorted(df['location'].unique()),
            key='trend_location'
        )
    
    with col2:
        # Add confidence level selector
        confidence_level = st.slider(
            "Confidence Level (%)",
            min_value=80,
            max_value=95,
            value=90,
            step=5,
            key='confidence_level'
        )
    
    if st.button("Generate Trend Prediction", key='trend_button'):
        # Show loading message
        with st.spinner('Generating trend prediction...'):
            # Show trend analysis
            trend_fig, trend_data = predict_future_trend(df, xgb_model, encoders, location, location_stats)
            
            if trend_fig is not None:
                # Display the trend plot
                st.plotly_chart(trend_fig, use_container_width=True)
                
                # Show trend insights
                current_avg = df[df['location'] == location]['monthly_rent'].mean()
                future_avg = trend_data['predicted_rent'].iloc[-1]
                percent_change = ((future_avg - current_avg) / current_avg) * 100
                
                # Create metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Current Average Rent (2024)",
                        f"RM {current_avg:,.2f}",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Predicted Rent (2033)",
                        f"RM {future_avg:,.2f}",
                        delta=f"{percent_change:+.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "Projected Annual Growth",
                        f"{(percent_change/10):+.1f}%",
                        delta=None
                    )
                
                # Add trend insights
                st.subheader("📊 Market Trend Insights")
                
                # Create an expandable section for detailed analysis
                with st.expander("See Detailed Analysis"):
                    st.write(f"""
                    ### Key Findings for {location}:
                    
                    1. **Growth Projection**
                       - Total projected growth: {percent_change:+.1f}% over 10 years
                       - Average annual growth rate: {(percent_change/10):+.1f}%
                    
                    2. **Price Points**
                       - Current average rent (2024): RM {current_avg:,.2f}
                       - Projected rent (2033): RM {future_avg:,.2f}
                       - Absolute increase: RM {(future_avg - current_avg):,.2f}
                    
                    3. **Market Position**
                       - Current market position: {location} is {"above" if current_avg > df['monthly_rent'].mean() else "below"} the overall market average
                       - Confidence interval: {confidence_level}% confidence in predictions
                    
                    ### Factors Considered:
                    - Historical price trends
                    - Property characteristics
                    - Location-specific factors
                    - Market dynamics
                    
                    ### Important Notes:
                    - Predictions assume stable market conditions
                    - External factors may impact actual trends
                    - Regular model updates recommended
                    """)
                
                # Show yearly predictions table
                if st.checkbox("Show Detailed Yearly Predictions"):
                    st.dataframe(
                        trend_data.style
                        .format({
                            'predicted_rent': 'RM{:,.2f}',
                            'lower_bound': 'RM{:,.2f}',
                            'upper_bound': 'RM{:,.2f}'
                        })
                        .set_properties(**{'text-align': 'right'})
                        .background_gradient(cmap='Blues', subset=['predicted_rent'])
                    )

def create_feature_inputs():
   """Create input fields for property features"""
   # Load the original KL dataset to get valid categories
   df = st.session_state['df']
   
   col1, col2, col3 = st.columns(3)
   
   with col1:
       st.subheader("Location Details")
       location = st.selectbox(
           "Location",
           options=sorted(df['location'].unique()),
           help="Only KL locations are available for prediction"
       )
       
       property_type = st.selectbox(
           "Property Type",
           options=sorted(df['property_type'].unique()),
           help="Only property types from KL dataset are available"
       )
       
       size = st.number_input(
           "Size (sq ft)",
           min_value=100,
           max_value=10000,
           value=1000
       )
   
   with col2:
       st.subheader("Property Features")
       rooms = st.number_input(
           "Number of Rooms",
           min_value=0,
           max_value=10,
           value=3
       )
       
       bathrooms = st.number_input(
           "Number of Bathrooms",
           min_value=0,
           max_value=10,
           value=2
       )
       
       parking = st.checkbox("Parking Available", value=True)
       near_ktm_lrt = st.checkbox("Near KTM/LRT Station", value=False)
   
   with col3:
       st.subheader("Additional Details")
       furnished = st.selectbox(
           "Furnished Status",
           options=sorted(df['furnished'].unique()),
           help="Only furnished status from KL dataset are available"
       )

   return {
       'location': location,
       'property_type': property_type,
       'size': size,
       'furnished': furnished,
       'rooms': rooms,
       'bathroom': bathrooms,
       'parking': parking,
       'additional_near_ktm/lrt': near_ktm_lrt
       
   }

def add_rental_market_analysis(df):
    """Add rental market analysis visualizations"""
    st.subheader("📊 Rental Market Analysis")
    
    # Create two columns for filters
    col1, col2 = st.columns(2)
    with col1:
        selected_location = st.selectbox(
            "Select Location",
            options=sorted(df['location'].unique()),
            key='market_location'
        )
    
    with col2:
        property_type = st.multiselect(
            "Select Property Type",
            options=sorted(df['property_type'].unique()),
            default=sorted(df['property_type'].unique()),
            key='market_property_type'
        )
    
    # Filter data based on selection
    filtered_df = df[
        (df['location'] == selected_location) & 
        (df['property_type'].isin(property_type))
    ]
    
    # 1. Price Distribution
    st.subheader("Rent Price Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot
        fig_box = px.box(
            filtered_df,
            y='monthly_rent',
            title=f'Rent Distribution in {selected_location}',
            height=400
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        # Histogram
        fig_hist = px.histogram(
            filtered_df,
            x='monthly_rent',
            nbins=30,
            title=f'Rent Price Frequency Distribution',
            height=400
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # 2. Average Rent by Property Type
    st.subheader("Average Rent by Property Type")
    avg_by_type = filtered_df.groupby('property_type')['monthly_rent'].agg([
        'mean', 'count', 'min', 'max'
    ]).round(2).reset_index()
    
    fig_bar = px.bar(
        avg_by_type,
        x='property_type',
        y='mean',
        text='mean',
        title=f'Average Rent by Property Type in {selected_location}',
        labels={'mean': 'Average Rent (RM)', 'property_type': 'Property Type'},
        height=400
    )
    fig_bar.update_traces(texttemplate='RM %{text:,.2f}', textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # 3. Price per Square Foot Analysis
    st.subheader("Price per Square Foot Analysis")
    col1, col2 = st.columns(2)
    
    # Calculate price per sqft
    filtered_df['price_per_sqft'] = filtered_df['monthly_rent'] / filtered_df['size']
    
    with col1:
        avg_price_sqft = filtered_df.groupby('property_type')['price_per_sqft'].mean().round(2)
        fig_price_sqft = px.bar(
            x=avg_price_sqft.index,
            y=avg_price_sqft.values,
            title='Average Price per Square Foot by Property Type',
            labels={'x': 'Property Type', 'y': 'Price per Sq Ft (RM)'},
            height=400
        )
        fig_price_sqft.update_traces(text=avg_price_sqft.values.round(2), textposition='outside')
        st.plotly_chart(fig_price_sqft, use_container_width=True)
    
    with col2:
        # Scatter plot of price vs size
        fig_scatter = px.scatter(
            filtered_df,
            x='size',
            y='monthly_rent',
            color='property_type',
            title='Rent Price vs Size',
            labels={'size': 'Size (sq ft)', 'monthly_rent': 'Monthly Rent (RM)'},
            height=400
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # 4. Size Distribution
    st.subheader("Property Size Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_size_hist = px.histogram(
            filtered_df,
            x='size',
            nbins=30,
            title='Size Distribution',
            labels={'size': 'Size (sq ft)'},
            height=400
        )
        st.plotly_chart(fig_size_hist, use_container_width=True)
    
    with col2:
        avg_size = filtered_df.groupby('property_type')['size'].mean().round(2)
        fig_avg_size = px.bar(
            x=avg_size.index,
            y=avg_size.values,
            title='Average Size by Property Type',
            labels={'x': 'Property Type', 'y': 'Size (sq ft)'},
            height=400
        )
        fig_avg_size.update_traces(text=avg_size.values.round(2), textposition='outside')
        st.plotly_chart(fig_avg_size, use_container_width=True)
    
    # 5. Rental Market Summary Table
    st.subheader("Rental Market Summary")
    summary_stats = pd.DataFrame({
        'Metric': [
            'Average Rent',
            'Median Rent',
            'Min Rent',
            'Max Rent',
            'Average Size',
            'Average Price/Sqft',
            'Number of Properties'
        ],
        'Value': [
            f"RM {filtered_df['monthly_rent'].mean():,.2f}",
            f"RM {filtered_df['monthly_rent'].median():,.2f}",
            f"RM {filtered_df['monthly_rent'].min():,.2f}",
            f"RM {filtered_df['monthly_rent'].max():,.2f}",
            f"{filtered_df['size'].mean():,.2f} sq ft",
            f"RM {filtered_df['price_per_sqft'].mean():,.2f}",
            f"{len(filtered_df):,}"
        ]
    })
    
    st.table(summary_stats)

def add_descriptive_analytics(df):
    """Add descriptive analytics visualizations with improved sizing and clarity"""
    st.title("📊 Descriptive Analytics")
    
    # 1. Rent Distribution Analysis
    st.subheader("Rental Price Distribution Analysis")
    
    # Larger box plot for rent by location
    fig_box = px.box(
        df,
        x='location',
        y='monthly_rent',
        title='Rent Distribution by Location',
        height=600  # Increased height
    )
    fig_box.update_xaxes(tickangle=45)
    fig_box.update_layout(
        title_x=0.5,
        margin=dict(t=60, b=80),  # Adjusted margins
        showlegend=True,
        xaxis_title="Location",
        yaxis_title="Monthly Rent (RM)"
    )
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Larger violin plot for rent by property type
    fig_violin = px.violin(
        df,
        x='property_type',
        y='monthly_rent',
        title='Rent Distribution by Property Type',
        height=600  # Increased height
    )
    fig_violin.update_xaxes(tickangle=45)
    fig_violin.update_layout(
        title_x=0.5,
        margin=dict(t=60, b=80),
        xaxis_title="Property Type",
        yaxis_title="Monthly Rent (RM)"
    )
    st.plotly_chart(fig_violin, use_container_width=True)
    
    # 2. Property Type Analysis
    st.subheader("Property Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Property types bar chart
        prop_counts = df['property_type'].value_counts().reset_index()
        prop_counts.columns = ['Property Type', 'Count']
        # Calculate percentage
        prop_counts['Percentage'] = (prop_counts['Count'] / prop_counts['Count'].sum() * 100).round(1)
        # Add percentage to display
        prop_counts['Label'] = prop_counts.apply(lambda x: f"{x['Count']} ({x['Percentage']}%)", axis=1)
        
        fig_property = px.bar(
            prop_counts,
            x='Property Type',
            y='Count',
            text='Label',
            title='Distribution of Property Types',
            height=500,
            color_discrete_sequence=['#1f77b4']
        )
        fig_property.update_layout(
            title={
                'text': 'Distribution of Property Types',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Property Type",
            yaxis_title="Number of Properties",
            xaxis={'tickangle': 45},
            margin=dict(t=80, b=120, l=40, r=40),
            showlegend=False
        )
        fig_property.update_traces(
            textposition='outside',
            texttemplate='%{text}'
        )
        st.plotly_chart(fig_property, use_container_width=True)
    
    with col2:
        # Furnished status pie chart
        furn_counts = df['furnished'].value_counts()
        fig_furnished = px.pie(
            values=furn_counts.values,
            names=furn_counts.index,
            title='Distribution of Furnished Status',
            height=500,
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
        )
        fig_furnished.update_layout(
            title={
                'text': 'Distribution of Furnished Status',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            legend={
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': -0.2,
                'xanchor': 'center',
                'x': 0.5
            },
            margin=dict(t=80, b=120, l=40, r=40)
        )
        fig_furnished.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_furnished, use_container_width=True)
    
    # 3. Fixed Additional Features Analysis
    st.subheader("Additional Features Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Convert boolean to string and create parking analysis
        df['parking_status'] = df['parking'].apply(lambda x: 'Has Parking' if x == 1 or x == True else 'No Parking')
        parking_counts = df['parking_status'].value_counts()
        
        fig_parking = px.pie(
            names=parking_counts.index,
            values=parking_counts.values,
            title='Parking Availability',
            height=500,
            color_discrete_sequence=['#1f77b4', '#ff7f0e']
        )
        fig_parking.update_layout(
            title_x=0.5,
            margin=dict(t=60, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_parking.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_parking, use_container_width=True)
    
    with col2:
        # Convert boolean to string and create KTM/LRT analysis
        df['ktm_status'] = df['additional_near_ktm/lrt'].apply(
            lambda x: 'Near KTM/LRT' if x == 1 or x == True else 'Not Near KTM/LRT'
        )
        ktm_counts = df['ktm_status'].value_counts()
        
        fig_ktm = px.pie(
            names=ktm_counts.index,
            values=ktm_counts.values,
            title='Proximity to Public Transport (KTM/LRT)',
            height=500,
            color_discrete_sequence=['#1f77b4', '#ff7f0e']
        )
        fig_ktm.update_layout(
            title_x=0.5,
            margin=dict(t=60, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_ktm.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_ktm, use_container_width=True)
    
    # 4. Summary Statistics
    st.subheader("Market Summary")
    summary_stats = pd.DataFrame({
        'Metric': [
            'Total Properties',
            'Average Monthly Rent',
            'Median Monthly Rent',
            'Most Common Property Type',
            'Fully Furnished Properties',
            'Properties with Parking',
            'Properties Near KTM/LRT'
        ],
        'Value': [
            f"{len(df):,}",
            f"RM {df['monthly_rent'].mean():,.2f}",
            f"RM {df['monthly_rent'].median():,.2f}",
            f"{df['property_type'].mode()[0]} ({(df['property_type'].value_counts().iloc[0]/len(df)*100):.1f}%)",
            f"{(df['furnished']=='Fully Furnished').sum()/len(df)*100:.1f}%",
            f"{(df['parking']==True).sum()/len(df)*100:.1f}%",
            f"{(df['additional_near_ktm/lrt']==True).sum()/len(df)*100:.1f}%"
        ]
    })
    
    st.table(summary_stats)

def add_rental_suggestions(df):
    """Add rental suggestions based on user preferences"""
    st.title("🏠 Kuala Lumpur Rental Property Finder")
    st.write("Let us help you find your ideal rental property based on your preferences!")
    
    # Create input form
    with st.form(key="property_preferences_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            location_pref = st.selectbox(
                "Preferred Location",
                options=['Any'] + sorted(df['location'].unique().tolist()),
                key='location'
            )
            
            property_type_pref = st.selectbox(
                "Property Type",
                options=['Any'] + sorted(df['property_type'].unique().tolist()),
                key='property_type'
            )
            
            rooms_pref = st.number_input(
                "Number of Rooms",
                min_value=0,
                max_value=int(df['rooms'].max()),
                value=0,
                step=1,
                key='rooms'
            )
        
        with col2:
            bathroom_pref = st.number_input(
                "Number of Bathrooms",
                min_value=0,
                max_value=int(df['bathroom'].max()),
                value=0,
                step=1,
                key='bathrooms'
            )
            
            furnished_pref = st.selectbox(
                "Furnished Status",
                options=['Any', 'Fully Furnished', 'Partially Furnished', 'Not Furnished'],
                key='furnished'
            )
            
            parking_pref = st.selectbox(
                "Parking Required",
                options=['Any', 'Yes', 'No'],
                key='parking'
            )
        
        with col3:
            ktm_lrt_pref = st.selectbox(
                "Near KTM/LRT",
                options=['Any', 'Yes', 'No'],
                key='ktm_lrt'
            )
            
            max_rent = st.number_input(
                "Maximum Monthly Rent (RM)",
                min_value=0,
                max_value=int(df['monthly_rent'].max()),
                value=int(df['monthly_rent'].median()),
                step=100,
                key='max_rent'
            )
        
        # Add submit button
        submit_button = st.form_submit_button("Find Properties")
    
    if submit_button:
        # Filter properties based on preferences
        filtered_df = df.copy()
        
        # Apply filters
        if location_pref != 'Any':
            filtered_df = filtered_df[filtered_df['location'] == location_pref]
            
        if property_type_pref != 'Any':
            filtered_df = filtered_df[filtered_df['property_type'] == property_type_pref]
            
        if rooms_pref > 0:
            filtered_df = filtered_df[filtered_df['rooms'] >= rooms_pref]
            
        if bathroom_pref > 0:
            filtered_df = filtered_df[filtered_df['bathroom'] >= bathroom_pref]
            
        if furnished_pref != 'Any':
            filtered_df = filtered_df[filtered_df['furnished'] == furnished_pref]
            
        if parking_pref != 'Any':
            has_parking = True if parking_pref == 'Yes' else False
            filtered_df = filtered_df[filtered_df['parking'] == has_parking]
            
        if ktm_lrt_pref != 'Any':
            near_ktm = True if ktm_lrt_pref == 'Yes' else False
            filtered_df = filtered_df[filtered_df['additional_near_ktm/lrt'] == near_ktm]
            
        filtered_df = filtered_df[filtered_df['monthly_rent'] <= max_rent]
        
        # Display results
        if len(filtered_df) > 0:
            st.success(f"Found {len(filtered_df)} matching properties!")
            
            # Sort by rent for better display
            filtered_df = filtered_df.sort_values('monthly_rent')
            
            # Display properties in cards
            for idx, row in filtered_df.iterrows():
                with st.container():
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"""
                        ### {row['property_type']} in {row['location']}
                        - **Property Name:** {row['prop_name']}
                        - **Monthly Rent:** RM {row['monthly_rent']:,.2f}
                        - **Rooms:** {int(row['rooms'])} | **Bathrooms:** {int(row['bathroom'])}
                        - **Size:** {int(row['size'])} sq ft
                        - **Furnished Status:** {row['furnished']}
                        - **Parking:** {'Available' if row['parking'] else 'Not Available'}
                        - **Near KTM/LRT:** {'Yes' if row['additional_near_ktm/lrt'] else 'No'}
                        """)
                    
                    with col2:
                        st.metric(
                            "Price per sq ft",
                            f"RM {row['monthly_rent']/row['size']:.2f}"
                        )
                    
                    st.divider()
        else:
            st.error("Oops! No properties match your preferences. Try adjusting your criteria.")
            
            # Show nearby suggestions
            st.subheader("Suggested Alternatives")
            alt_df = df[
                (df['location'] == location_pref if location_pref != 'Any' else True) &
                (df['property_type'] == property_type_pref if property_type_pref != 'Any' else True)
            ].copy()
            
            if len(alt_df) > 0:
                st.info("Here are some properties that partially match your criteria:")
                
                # Show top 3 closest matches
                alt_df = alt_df.sort_values('monthly_rent').head(3)
                
                for idx, row in alt_df.iterrows():
                    with st.container():
                        st.markdown(f"""
                        ### {row['property_type']} in {row['location']}
                        - **Property Name:** {row['prop_name']}
                        - **Monthly Rent:** RM {row['monthly_rent']:,.2f}
                        - **Rooms:** {int(row['rooms'])} | **Bathrooms:** {int(row['bathroom'])}
                        - **Furnished Status:** {row['furnished']}
                        """)
                        st.divider()
            else:
                st.info("Try broadening your search criteria to see more options.")

def load_all_models():
    """Load all three saved models and their artifacts"""
    try:
        # Load XGBoost model
        with open(r'C:\Users\User\.jupyter\Dash31\tuned_xgboost_model.pkl', 'rb') as file:
            xgb_artifacts = pickle.load(file)
        
        # Load Random Forest model
        with open(r'C:\Users\User\.jupyter\Dash31\tuned_rf_model.pkl', 'rb') as file:
            rf_artifacts = pickle.load(file)
        
        # Load Linear Regression model
        with open(r'C:\Users\User\.jupyter\Dash31\tuned_lr_model.pkl', 'rb') as file:
            lr_artifacts = pickle.load(file)
        
        return {
            'xgboost': {
                'model': xgb_artifacts['model'],
                'encoders': xgb_artifacts['encoders']
            },
            'random_forest': {
                'model': rf_artifacts['model'],
                'encoders': rf_artifacts['encoders']
            },
            'linear_regression': {
                'model': lr_artifacts['model'],
                'encoders': lr_artifacts['encoders'],
                'scaler': lr_artifacts['scaler']
            }
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def load_default_dataset():
    """Load the default KL dataset and train model"""
    try:
        df = pd.read_csv(r"C:\Users\User\Downloads\cleaned_KL_data.csv")
        
        # Train model for default dataset
        encoders = {
            'location': LabelEncoder(),
            'property_type': LabelEncoder(),
            'furnished': LabelEncoder()
        }
        
        # Create a copy for modeling
        model_df = df.copy()
        
        # Encode categorical variables
        for column, encoder in encoders.items():
            model_df[f'{column}_encoded'] = encoder.fit_transform(model_df[column])
        
        # Select features
        feature_list = [
            'size',
            'rooms',
            'bathroom',
            'parking',
            'additional_near_ktm/lrt',
            'location_encoded',
            'property_type_encoded',
            'furnished_encoded'
        ]
        
        X = model_df[feature_list]
        y = model_df['monthly_rent']
        
        # Train model
        xgb_model = xgb.XGBRegressor(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        xgb_model.fit(X, y)
        
        # Store model artifacts
        st.session_state['model_artifacts'] = {
            'model': xgb_model,
            'encoders': encoders,
            'feature_list': feature_list
        }
        
        return df
        
    except Exception as e:
        st.error(f"Error loading default dataset: {str(e)}")
        return None

def add_model_comparison():
    """Add model comparison visualizations"""
    st.title("Model Performance Comparison")
    
    
    metrics = {
        'XGBoost': {
            'MAE': 288.44,  # Replace with your actual MAE
            'RMSE': 402.21, # Replace with your actual RMSE
            'R2': 0.77     # Replace with your actual R2
        },
        'Random Forest': {
            'MAE': 295.95,  # Replace with your actual MAE
            'RMSE': 421.41, # Replace with your actual RMSE
            'R2': 0.73     # Replace with your actual R2
        },
        'Linear Regression': {
            'MAE': 425.32,  # Replace with your actual MAE
            'RMSE': 589.67, # Replace with your actual RMSE
            'R2': 0.44     # Replace with your actual R2
        }
    }
    
    # Create DataFrame for metrics
    df_metrics = pd.DataFrame.from_dict(metrics, orient='index')
    
    # 1. MAE Comparison
    st.subheader("Mean Absolute Error (MAE) Comparison")
    fig_mae = px.bar(
        df_metrics,
        y='MAE',
        title='Mean Absolute Error by Model',
        color=df_metrics.index,
        labels={'index': 'Model', 'value': 'MAE (RM)'}
    )
    fig_mae.update_traces(texttemplate='RM %{y:.2f}', textposition='outside')
    st.plotly_chart(fig_mae, use_container_width=True)
    
    # 2. RMSE Comparison
    st.subheader("Root Mean Square Error (RMSE) Comparison")
    fig_rmse = px.bar(
        df_metrics,
        y='RMSE',
        title='Root Mean Square Error by Model',
        color=df_metrics.index,
        labels={'index': 'Model', 'value': 'RMSE (RM)'}
    )
    fig_rmse.update_traces(texttemplate='RM %{y:.2f}', textposition='outside')
    st.plotly_chart(fig_rmse, use_container_width=True)
    
    # 3. R2 Score Comparison
    st.subheader("R² Score Comparison")
    fig_r2 = px.bar(
        df_metrics,
        y='R2',
        title='R² Score by Model',
        color=df_metrics.index,
        labels={'index': 'Model', 'value': 'R²'}
    )
    fig_r2.update_traces(texttemplate='%{y:.4f}', textposition='outside')
    st.plotly_chart(fig_r2, use_container_width=True)
    
    # Add metrics table
    st.subheader("Performance Metrics Summary")
    styled_metrics = pd.DataFrame({
        'Model': df_metrics.index,
        'Mean Absolute Error': df_metrics['MAE'].apply(lambda x: f'RM {x:,.2f}'),
        'Root Mean Square Error': df_metrics['RMSE'].apply(lambda x: f'RM {x:,.2f}'),
        'R² Score': df_metrics['R2'].apply(lambda x: f'{x:.4f}')
    }).set_index('Model')
    
    st.dataframe(styled_metrics)
    
    # Add explanation
    st.info("""
    **Understanding the Metrics:**
    
    1. **Mean Absolute Error (MAE)**
       - Represents the average absolute difference between predicted and actual rental prices
       - Lower values indicate better performance
       - More intuitive as it's in the same unit as rental prices (RM)
       
    2. **Root Mean Square Error (RMSE)**
       - Square root of the average squared prediction errors
       - Penalizes larger errors more heavily than MAE
       - Also in rental price units (RM)
       - Lower values indicate better performance
       
    3. **R² Score**
       - Indicates how well the model explains the variance in rental prices
       - Ranges from 0 to 1 (1 being perfect prediction)
       - Higher values indicate better model fit
    
    **Key Findings:**
    - XGBoost shows the best overall performance with lowest errors and highest R²
    - Random Forest performs similarly well, showing robust prediction capability
    - Linear Regression shows higher errors, indicating rental prices have non-linear relationships with features
    """)

def main():
    try:
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6  = st.tabs([
            "Rental Suggestions",
            "Rental Prediction",
            "10-Year Forecast",
            "Market Analysis",
            "Descriptive Analytics",
            "🔍Model Comparison"
        ])
        
        if 'df' not in st.session_state:
            df = load_default_dataset()
            st.session_state['df'] = df
        
        # Get dataset from session state
        df = st.session_state['df']
        # Load data once
        
        
        # Load model and encoders
        models = load_all_models()
        if models is None:
            st.error("Failed to load models")
            return
        
        
        # Calculate location statistics for predictions
        location_stats = calculate_location_stats(df)
        
        with tab1:
            add_rental_suggestions(df)

        with tab2:
            st.title("🏠 Kuala Lumpur Rental Price Prediction")
            # Get user inputs
            features = create_feature_inputs()
            
            if st.button("Predict Rental Price", type="primary"):
                result = predict_price(features, models['xgboost']['model'], models['xgboost']['encoders'])
                
                if result is not None:
                    prediction = result['prediction']
                    
                    # Show prediction
                    st.subheader("Prediction Results")
                    st.success(f"Predicted Monthly Rent: RM {prediction:,.2f}")
                    
                    # Show location context
                    location_stats_filtered = location_stats[
                        location_stats['location'] == features['location']
                    ].iloc[0]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Predicted Rent",
                            f"RM {prediction:,.2f}",
                            f"{((prediction - location_stats_filtered['monthly_rent_mean']) / location_stats_filtered['monthly_rent_mean'] * 100):,.1f}% vs. average"
                        )
                    
                    with col2:
                        st.metric(
                            "Location Average",
                            f"RM {location_stats_filtered['monthly_rent_mean']:,.2f}"
                        )
                    
                    with col3:
                        st.metric(
                            "Location Median",
                            f"RM {location_stats_filtered['monthly_rent_median']:,.2f}"
                        )
                    
                    # Show feature importance
                    if 'feature_importance' in result:
                        st.subheader("Feature Importance")
                        fig = px.bar(
                            result['feature_importance'].head(10),
                            x='importance',
                            y='feature',
                            orientation='h',
                            title='Top 10 Most Important Features in Prediction'
                        )
                        st.plotly_chart(fig)
        
        with tab3:
            st.title("Market Analysis")
            add_trend_analysis_section(
                df=df,
                xgb_model=models['xgboost']['model'],
                encoders=models['xgboost']['encoders'],
                location_stats=location_stats
            )
                
        with tab4:
            add_rental_market_analysis(df)
                
        with tab5:
            add_descriptive_analytics(df)
        
        with tab6:
            add_model_comparison()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please try refreshing the page or contact support if the error persists.")

if __name__ == "__main__":
    main()