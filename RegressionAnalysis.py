import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.express as px

# Load the data
regression_file = pd.read_csv(r"D:\FYP\plots_vs_prices_regression\property2.tsv", sep='\t')

# Initialize a DataFrame to store predictions
predictions = pd.DataFrame(columns=['Location', 'Year', 'Predicted_Price', 'MSE'])

# Generate predictions and evaluate model
unique_locations = regression_file['Location'].unique()
for location in unique_locations:
    temp_df = regression_file[regression_file['Location'] == location].reset_index(drop=True)

    # Model training
    model = LinearRegression()
    model.fit(temp_df['Year'].to_numpy().reshape(-1, 1), temp_df['Prices'].to_numpy())
    
    # Predict for the year 2030 and calculate Mean Squared Error (MSE)
    predict_y = model.predict(np.array([[2030]]))
    mse = mean_squared_error(temp_df['Prices'], model.predict(temp_df['Year'].to_numpy().reshape(-1, 1)))

    # Create new record for predictions
    new_record = pd.DataFrame({'Location': [location], 'Year': [2030], 'Predicted_Price': [round(predict_y[0], 2)], 'MSE': [round(mse, 2)]})
    
    # Append using pd.concat, ensuring no empty entries
    predictions = pd.concat([predictions, new_record], ignore_index=True)

# Save the predictions to CSV
predictions.to_csv(r'D:\FYP\predictions_vs_mse.csv', index=False, header=True)

# Merge predictions with original data for plotting
extended_data = pd.concat([regression_file, predictions.rename(columns={'Predicted_Price': 'Prices'})])

# Create interactive plot with Plotly
def interactive_temporal_plot():
    fig = px.line(extended_data, x='Location', y='Prices', color='Year', title="Property Price Trends by Location", 
                  color_discrete_sequence=px.colors.qualitative.Set1)
    fig.update_layout(xaxis_title="Location", yaxis_title="Prices [per Sq. ft]", legend_title="Year")
    fig.show()

# Function for static Seaborn plot without annotations
def temporal_plot_with_mse(log_y=True):
    plt.figure(figsize=(18, 10))
    sns.lineplot(data=extended_data, x='Location', y='Prices', hue='Year', palette='Set1')
    plt.yscale('log' if log_y else 'linear')
    plt.ylabel('Prices [per Sq. ft]')
    plt.xlabel('Location')
    plt.title('Property Prices by Location Over Time')
    plt.xticks(rotation=90)
    
    plt.tight_layout()
    fig_title = 'location_vs_prices_mse_log' if log_y else 'location_vs_prices_mse'
    plt.savefig(rf'D:\FYP\{fig_title}.png', dpi=350)
    plt.show()

# Display plots
interactive_temporal_plot()
temporal_plot_with_mse(log_y=True)
temporal_plot_with_mse(log_y=False)
