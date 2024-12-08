import joblib
import pandas as pd
import numpy as np
from shiny import App, render, ui, reactive
import matplotlib.pyplot as plt

# Load the pre-trained model and data
model = joblib.load('/Users/jaydenmurata/Desktop/fraud/fraud_prediction.joblib')
data = pd.read_csv('/Users/jaydenmurata/Desktop/fraud/balanced_bigfraud.csv')

# Identify numeric columns
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

# Remove target column if present
if 'target' in numeric_columns:
    numeric_columns.remove('target')

# Prepare the app UI
app_ui = ui.page_sidebar(
    ui.sidebar(
        # Dynamic feature selection based on numeric columns
        ui.input_select(
            id="feature1", 
            label="Select First Feature", 
            choices=numeric_columns
        ),
        ui.input_select(
            id="feature2", 
            label="Select Second Feature", 
            choices=numeric_columns
        ),
        ui.input_slider(
            id="feature1_range", 
            label="Feature 1 Range",
            min=float(data[numeric_columns[0]].min()),
            max=float(data[numeric_columns[0]].max()),
            value=(float(data[numeric_columns[0]].min()), float(data[numeric_columns[0]].max()))
        ),
        ui.input_slider(
            id="feature2_range", 
            label="Feature 2 Range",
            min=float(data[numeric_columns[1]].min()),
            max=float(data[numeric_columns[1]].max()),
            value=(float(data[numeric_columns[1]].min()), float(data[numeric_columns[1]].max()))
        ),
        ui.input_action_button(id="predict_btn", label="Predict Fraud")
    ),
    
    # Main panel content
    ui.card(
        ui.card_header("Fraud Prediction Results"),
        ui.output_text_verbatim("prediction_result")
    ),
    ui.card(
        ui.card_header("Feature Distribution"),
        ui.output_plot("fraud_plot")
    )
)

# Define the server logic
def server(input, output, session):
    @output
    @render.text
    def prediction_result():
        # Check if prediction button is clicked
        if not input.predict_btn():
            return "Click 'Predict Fraud' to see results"
        
        # Create a sample data point using selected features
        sample_data = pd.DataFrame({
            input.feature1(): [np.mean([input.feature1_range()[0], input.feature1_range()[1]])],
            input.feature2(): [np.mean([input.feature2_range()[0], input.feature2_range()[1]])]
        })
        
        # Make prediction
        prediction = model.predict(sample_data)
        proba = model.predict_proba(sample_data)
        
        return f"""
Fraud Prediction Results:
-------------------------
Selected Features: {input.feature1()} and {input.feature2()}
Prediction: {'Fraud' if prediction[0] == 1 else 'Not Fraud'}
Probability of Fraud: {proba[0][1]:.2%}
        """
    
    @output
    @render.plot
    def fraud_plot():
        # Determine target column
        target_col = 'target' if 'target' in data.columns else None
        
        plt.figure(figsize=(10, 6))
        if target_col:
            plt.scatter(
                data[data[target_col] == 0][input.feature1()], 
                data[data[target_col] == 0][input.feature2()], 
                label='Non-Fraud', 
                alpha=0.5
            )
            plt.scatter(
                data[data[target_col] == 1][input.feature1()], 
                data[data[target_col] == 1][input.feature2()], 
                label='Fraud', 
                color='red', 
                alpha=0.5
            )
        else:
            plt.scatter(
                data[input.feature1()], 
                data[input.feature2()], 
                alpha=0.5
            )
        
        plt.xlabel(input.feature1())
        plt.ylabel(input.feature2())
        plt.title('Feature Distribution')
        plt.legend()
        return plt

# Create the Shiny app
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()