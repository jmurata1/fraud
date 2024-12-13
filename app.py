from shiny import App, render, ui, reactive
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load the pre-trained model and data
model = joblib.load('/Users/jaydenmurata/Desktop/fraud/fraud_prediction.joblib')
data = pd.read_csv('/Users/jaydenmurata/Desktop/fraud/balanced_bigfraud.csv')

# Define feature groups
predictors = ["category", "job_category", "distance_miles", "hour_of_day", "amt"]
contin = ["distance_miles", "amt"]
categorical = ["category", "job_category", "hour_of_day"]

# Define the optimal threshold
FRAUD_THRESHOLD = 0.30

# Define custom color scheme
COLORS = {
    'background': '#e8f5e9',  # Light green background
    'primary': '#2e7d32',     # Dark green for primary elements
    'secondary': '#81c784',   # Medium green for secondary elements
    'fraud': '#d32f2f',       # Red for fraud
    'safe': '#388e3c'         # Green for safe
}

app_ui = ui.page_fluid(
    ui.tags.style(
        """
        body { background-color: #e8f5e9; }
        .card { background-color: white; border: 1px solid #81c784; }
        .btn-primary { background-color: #2e7d32 !important; border-color: #2e7d32 !important; }
        .nav-tabs .nav-link.active { color: #2e7d32 !important; }
        .nav-tabs .nav-link { color: #81c784 !important; }
        """
    ),
    ui.h2("Fraud Detection Dashboard", style="color: #2e7d32;"),
    ui.navset_tab(
        ui.nav_panel("Prediction",
            ui.input_select(
                "input_category",
                "Select category:",
                choices=sorted(data['category'].unique())
            ),
            ui.input_select(
                "input_job_category",
                "Select job category:",
                choices=sorted(data['job_category'].unique())
            ),
            ui.input_select(
                "input_hour_of_day",
                "Select hour of day:",
                choices=list(range(24))
            ),
            ui.input_numeric(
                "input_distance_miles",
                "Enter distance (miles):",
                value=round(float(data['distance_miles'].mean()), 2),
                step=0.01
            ),
            ui.input_numeric(
                "input_amt",
                "Enter amount ($):",
                value=round(float(data['amt'].mean()), 2),
                step=0.01
            ),
            ui.input_action_button("predict_btn", "Predict Fraud"),
            ui.output_text("prediction_result")
        ),
        ui.nav_panel("Data Analysis",
            ui.input_select(
                "plot_feature",
                "Choose feature to analyze:",
                choices=predictors
            ),
            ui.output_plot("fraud_distribution"),
            ui.output_plot("fraud_rate")
        )
    )
)

def server(input, output, session):
    # Create a reactive value to store button clicks
    clicks = reactive.Value(0)
    
    @reactive.Effect
    @reactive.event(input.predict_btn)
    def _():
        clicks.set(clicks.get() + 1)

    @output
    @render.text
    @reactive.event(input.predict_btn)
    def prediction_result():
        if clicks.get() == 0:
            return "Click 'Predict Fraud' to see results"
        
        try:
            input_data = {
                'category': [input.input_category()],
                'job_category': [input.input_job_category()],
                'hour_of_day': [int(input.input_hour_of_day())],
                'distance_miles': [round(float(input.input_distance_miles()), 2)],
                'amt': [round(float(input.input_amt()), 2)]
            }
            
            sample = pd.DataFrame(input_data)
            proba = model.predict_proba(sample)
            prediction = (proba[0][1] > FRAUD_THRESHOLD).astype(int)
            
            result_header = "🚨 FRAUD DETECTED! 🚨" if prediction == 1 else "✅ Transaction Appears Safe"
            
            return f"""
================================================================
{result_header}
================================================================

Transaction Details:
- Category: {input_data['category'][0]}
- Job Category: {input_data['job_category'][0]}
- Hour of Day: {input_data['hour_of_day'][0]}
- Distance: {input_data['distance_miles'][0]} miles
- Amount: ${input_data['amt'][0]:,.2f}

Risk Assessment:
- Fraud Probability: {proba[0][1]:.1%}
- Threshold Used: {FRAUD_THRESHOLD:.1%}

Recommendation:
{'⚠️  This transaction requires immediate review!' if prediction == 1 else '✓  This transaction can proceed normally.'}
"""
        except Exception as e:
            return f"Error making prediction: {str(e)}\nInput data: {input_data}"

    # Replace just the plot functions in your code with these updated versions:

    @output
    @render.plot
    def fraud_distribution():
        if input.plot_feature() == 'job_category':
            fig = plt.figure(figsize=(15, 10))
        else:
            fig = plt.figure(figsize=(12, 8))
            
        feature = input.plot_feature()
        
        if feature in categorical:
            if feature in ['category', 'job_category']:
                # Filter out "Other" category for both category and job_category
                filtered_data = data[data[feature] != 'Other']
                plt.subplots_adjust(bottom=0.3)
            else:
                filtered_data = data
                
            categories = sorted(filtered_data[feature].unique())
            safe_counts = [len(filtered_data[(filtered_data[feature]==cat) & (filtered_data['is_fraud']==0)]) for cat in categories]
            fraud_counts = [len(filtered_data[(filtered_data[feature]==cat) & (filtered_data['is_fraud']==1)]) for cat in categories]
            
            x = np.arange(len(categories))
            width = 0.35
            
            plt.bar(x - width/2, safe_counts, width, label='Safe', color=COLORS['safe'])
            plt.bar(x + width/2, fraud_counts, width, label='Fraud', color=COLORS['fraud'])
            
            if feature == 'hour_of_day':
                plt.xticks(x, range(24), rotation=0)
            else:
                plt.xticks(x, categories, rotation=90, ha='center')
            
            plt.legend()
            plt.title(f'Distribution by {feature}')
            plt.xlabel(feature.replace('_', ' ').title())
            plt.ylabel('Count')
            
        else:
            if feature == 'amt':
                filtered_data = data[data['amt'] <= 1500]
                plt.hist(filtered_data[filtered_data['is_fraud']==0][feature], 
                        bins=30, alpha=0.7, label='Safe', color=COLORS['safe'])
                plt.hist(filtered_data[filtered_data['is_fraud']==1][feature], 
                        bins=30, alpha=0.7, label='Fraud', color=COLORS['fraud'])
                plt.title(f'Distribution of {feature} (≤ $1,500)')
                plt.xlabel('Amount ($)')
            else:
                plt.hist(data[data['is_fraud']==0][feature], 
                        bins=30, alpha=0.7, label='Safe', color=COLORS['safe'])
                plt.hist(data[data['is_fraud']==1][feature], 
                        bins=30, alpha=0.7, label='Fraud', color=COLORS['fraud'])
                plt.title(f'Distribution of {feature}')
                plt.xlabel(feature.replace('_', ' ').title())
            plt.ylabel('Count')
            plt.legend()
        
        if feature not in ['category', 'job_category']:
            plt.tight_layout()
        return fig

    @output
    @render.plot
    def fraud_rate():
        if input.plot_feature() == 'job_category':
            fig = plt.figure(figsize=(15, 10))
        else:
            fig = plt.figure(figsize=(12, 8))
            
        feature = input.plot_feature()
        
        if feature in categorical:
            if feature in ['category', 'job_category']:
                # Filter out "Other" category for both category and job_category
                filtered_data = data[data[feature] != 'Other']
                fraud_rate = filtered_data.groupby(feature)['is_fraud'].mean().sort_values(ascending=False)
                plt.subplots_adjust(bottom=0.3)
            elif feature == 'hour_of_day':
                fraud_rate = data.groupby(feature)['is_fraud'].mean()
                fraud_rate = fraud_rate.reindex(range(24))
            else:
                fraud_rate = data.groupby(feature)['is_fraud'].mean().sort_values(ascending=False)
            
            x = np.arange(len(fraud_rate))
            plt.bar(x, fraud_rate.values, color=COLORS['secondary'])
            
            if feature == 'hour_of_day':
                plt.xticks(x, range(24), rotation=0)
            else:
                plt.xticks(x, fraud_rate.index, rotation=90, ha='center')
                
            plt.title(f'Fraud Rate by {feature}')
            plt.xlabel(feature.replace('_', ' ').title())
            
        else:
            if feature == 'amt':
                filtered_data = data[data['amt'] <= 1500].copy()
                percentiles = np.percentile(filtered_data[feature], np.linspace(0, 100, 11))
                percentiles = np.unique(percentiles)
                bin_labels = [f'${percentiles[i]:.0f}-${percentiles[i+1]:.0f}'
                            for i in range(len(percentiles)-1)]
                filtered_data['bins'] = pd.cut(filtered_data[feature], 
                                            bins=percentiles,
                                            labels=bin_labels,
                                            include_lowest=True)
                fraud_rate = filtered_data.groupby('bins')['is_fraud'].mean()
                plt.xlabel('Amount Range')
            else:
                percentiles = np.percentile(data[feature], np.linspace(0, 100, 11))
                percentiles = np.unique(percentiles)
                bin_labels = [f'{percentiles[i]:.2f}-{percentiles[i+1]:.2f}'
                            for i in range(len(percentiles)-1)]
                data['bins'] = pd.cut(data[feature], 
                                    bins=percentiles,
                                    labels=bin_labels,
                                    include_lowest=True)
                fraud_rate = data.groupby('bins')['is_fraud'].mean()
                plt.xlabel(f'{feature.replace("_", " ").title()} Range')
            
            x = np.arange(len(fraud_rate))
            plt.bar(x, fraud_rate.values, color=COLORS['secondary'])
            plt.xticks(x, fraud_rate.index, rotation=45, ha='right')
            plt.title(f'Fraud Rate by {feature} Ranges')
        
        plt.ylabel('Fraud Rate')
        
        if feature not in ['category', 'job_category']:
            plt.tight_layout()
        return fig

app = App(app_ui, server)