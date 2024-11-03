from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained machine learning model from a pickle file
with open('models/random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('form.html', prediction=None)  # Pass 'None' as initial prediction

# Define the route to handle form submission
@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Check if a file was uploaded
        if 'file' not in request.files:
            return "No file part in the request"
        
        file = request.files['file']

        # If the user does not select a file, the browser may submit an empty file
        if file.filename == '':
            return "No file selected"

        # Read the CSV file into a DataFrame
        try: 
            data = pd.read_csv(file)
        except Exception as e:
            return f"Error: {str(e)}"
        
        # Use pd.get_dummies to one-hot encode the 'industry' column and any other categorical columns
        try:
            data = pd.get_dummies(data, columns=['industry'], drop_first=True)
        except Exception as e:
            return f"Error: {str(e)}"
        
        # Columns to drop
        columns_to_drop = ['customer_number','stage_grouped','max_closed_date','cloud_revenue','total_opp_amount','country']

        # Split the data into features and target for classification
        X_class = data.drop(columns=columns_to_drop)

        # Standardize the features
        scaler = StandardScaler()
        X_class = scaler.fit_transform(X_class)
        
        # Ensure all categorical columns during training are present in the data
        expected_columns = ['employees', 'ee_count', 'contacts_count', 'emb_world_attendees',
       'email_clicks', 'a365_new_wb_visits', 'a365_rest_wb_visits',
       'total_webinars_attended', 'assembly_app_events', 'comms_days_used',
       'libs_events', 'libs_days_used', 'manufacturing_events', 'mcad_events',
       'multi_cad_events', 'part_request_events', 'share_events',
       'tasks_events', 'version_control_events', 'web_review_events',
       'industry_Aerospace', 'industry_Aerospace & Defense',
       'industry_Automotive', 'industry_Chemicals', 'industry_Communications',
       'industry_Component OEM', 'industry_Construction',
       'industry_Consulting', 'industry_Consumer Electronics',
       'industry_Electronic Component and Semiconductor',
       'industry_Electronics', 'industry_Engineering',
       'industry_Hospitals and Healthcare',
       'industry_Industrial Manufacturing and Services',
       'industry_Industrial/Instrumentation', 'industry_Machinery',
       'industry_Manufacturing', 'industry_Maritime',
       'industry_Medical Device', 'industry_Medical Devices',
       'industry_Military/Defense', 'industry_Non-Traditional Sectors',
       'industry_Other', 'industry_Retail',
       'industry_Service Bureau / Engineering Services', 'industry_Technology',
       'industry_Telecommunications']
        for column in expected_columns:
            if column not in data.columns:
                data[column] = 0
    
        # Reorder the columns
        X_class = data[expected_columns]
        
        # Update 0 values in columns that start with industry to False
        for column in X_class.columns:
            if column.startswith('industry'):
                X_class[column] = X_class[column].astype(bool)

        # Make predictions
        predictions = model.predict(X_class)
        probabilities = model.predict_proba(X_class)

        # Add the predictions and probabilities to the DataFrame
        data['Prediction'] = predictions
        data['Probability'] = probabilities.max(axis=1)
        # Set threshold for classification
        threshold = 0.6
        data['Prediction_Threshold'] = data['Probability'].apply(lambda x: 1 if x > threshold else 0)
        # Reorder the columns
        data = data[['Prediction', 'Probability', 'Prediction_Threshold','customer_number']]
    
         # Manually create a styled HTML table using Tailwind CSS
        table_html = "<table class='min-w-full divide-y divide-gray-200'>"
        table_html += "<thead class='bg-gray-50'><tr>"
        for column in data.columns:
            table_html += f"<th class='px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider'>{column}</th>"
        table_html += "</tr></thead><tbody class='bg-white divide-y divide-gray-200'>"
        for _, row in data.iterrows():
            table_html += "<tr>"
            for cell in row:
                table_html += f"<td class='px-6 py-4 whitespace-nowrap text-sm text-gray-900'>{cell}</td>"
            table_html += "</tr>"
        table_html += "</tbody></table>"

        # Render the results.html template and pass the table HTML
        return render_template('results.html', table_html=table_html)

    except Exception as e:
        return f"Error: {str(e)}"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
