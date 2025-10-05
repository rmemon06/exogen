import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os
from flask import Flask, request, render_template_string, url_for, flash, redirect
import matplotlib
matplotlib.use('Agg') # because of the images we generate for the webapp it would crash without this 
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

#----------------------------------------------------model training-----------------------------------------------------#

MODEL_FILE = 'exoplanet_model.joblib' # File to save/load the trained model
DATA_FILE = 'kepler_objects_of_interest.csv'

def train_and_save_model():
    """
    Trains, evaluates, and saves the Random Forest model, imputer, and evaluation artifacts.
    """
    print("Training a new model, as no saved model was found")
    
    df = pd.read_csv(DATA_FILE, comment="#")#using data frame to load the csv file

    # Select relevant columns as the features and target
    columns_to_keep = [
        'koi_period', 'koi_ror', 'koi_prad', 'koi_dor', 'koi_depth',
        'koi_model_snr', 'koi_steff', 'koi_slogg', 'koi_srad', 
        'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 
        'koi_impact', 'koi_duration', 'koi_disposition'
    ]
    processed_data = df[columns_to_keep].copy()
    
    # Map target variable to numbers before splitting, i think this is conventional/necessary
    processed_data["koi_disposition"] = processed_data["koi_disposition"].map({
        'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0
    })
    
    # Drop rows where the target variable itself is missing cuz we cant train on or use those at all
    processed_data.dropna(subset=['koi_disposition'], inplace=True)

    #the features are X and the target variable is Y
    X = processed_data.drop("koi_disposition", axis=1)
    Y = processed_data["koi_disposition"]
    
    # imputation meaning filling missing values using median
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns) # Convert back to a DataFrame

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)# 0.2 means 20% of data is used for testing

    # hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42), 
        param_grid, 
        cv=3, 
        scoring='accuracy', 
        n_jobs=-1
    )
    
    print("Tuning hyperparameters... this might take a few minutes.")
    grid_search.fit(X_train, Y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")

    # --- Evaluate the model and generate charts ---
    print("Evaluating model and generating visualizations...")
    Y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    matrix = confusion_matrix(Y_test, Y_pred)
    
    #generate the comnfusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED'], yticklabels=['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED'])
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Model Confusion Matrix')
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    cm_img_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    plt.close()

    #and generate the importance of features graph
    importances = best_model.feature_importances_
    feature_names = X.columns
    feature_importance_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importance_series, y=feature_importance_series.index, palette='viridis')
    plt.xlabel('Importance Score'); plt.ylabel('Features'); plt.title('Feature Importance in Exoplanet Detection')
    plt.tight_layout()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    fi_img_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    plt.close()
    
    # now that we have the best model, we save it along with the imputer and feature names in a dictionary called artifacts
    artifacts = {
        'model': best_model, 'imputer': imputer, 'features': X.columns,
        'accuracy': accuracy, 'confusion_matrix_img': cm_img_base64,
        'feature_importance_img': fi_img_base64
    }
    joblib.dump(artifacts, MODEL_FILE)
    print(f"Model and artifacts saved to {MODEL_FILE}")
    return artifacts

#----------------------------------------------------web app-----------------------------------------------------# 
app = Flask(__name__)
app.secret_key = 'your_super_secret_key' # Needed for flash messages
artifacts = None

# Check if the model file exists, if not, train it
if not os.path.exists(MODEL_FILE):
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: The data file '{DATA_FILE}' was not found.")
        exit()
    artifacts = train_and_save_model()
else:
    print(f"Loading existing artifacts from {MODEL_FILE}...")
    artifacts = joblib.load(MODEL_FILE)

print("Loading Kepler dataset for lookups...")
full_df = pd.read_csv(DATA_FILE, comment="#")
DISPOSITION_MAP = {2: 'CONFIRMED', 1: 'CANDIDATE', 0: 'FALSE POSITIVE'}

# --- Unified HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Exogen - {{ title }}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .dropdown-content { display: none; position: absolute; background-color: #1F2937; min-width: 160px; box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2); z-index: 1; border-radius: 0.5rem; }
        .dropdown:hover .dropdown-content { display: block; }
        body { font-family: 'Inter', sans-serif; }
        .alert { padding: 1rem; margin-bottom: 1rem; border-radius: 0.5rem; }
        .alert-success { background-color: #059669; color: #D1FAE5; }
        .alert-error { background-color: #B91C1C; color: #FEE2E2; }
    </style>
    <link rel="stylesheet" href="https://rsms.me/inter/inter.css">
</head>
<body class="bg-gray-900 text-white">
    <div class="container mx-auto p-4">
        <header class="flex justify-between items-center mb-8">
            <div class="dropdown relative">
                <button class="text-3xl font-bold tracking-wider cursor-pointer flex items-center">
                    <span>EXOGEN</span>
                    <svg class="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                </button>
                <div class="dropdown-content pt-4 pb-2">
                    <a href="{{ url_for('home') }}" class="block px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white">Home</a>
                    <a href="{{ url_for('about') }}" class="block px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white">About</a>
                    <a href="{{ url_for('data') }}" class="block px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white">Data & Insights</a>
                    <a href="{{ url_for('predict_csv') }}" class="block px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white">Predict from CSV</a>
                </div>
            </div>
        </header>
        <main>
            {% with messages = get_flashed_messages(with_categories=true) %}
                {% if messages %}
                    {% for category, message in messages %}
                        <div class="alert alert-{{ category }}">{{ message | safe }}</div>
                    {% endfor %}
                {% endif %}
            {% endwith %}

            {% if page == 'home' %}
            <div class="flex justify-center mb-8">
                <img src="{{ url_for('static', filename='spaceappslogo.png') }}" alt="Exoplanet Illustration" class="w-52 h-52 rounded-full object-cover shadow-2xl border-4 border-gray-700">
            </div>
            <div class="bg-gray-800 p-8 rounded-lg shadow-2xl max-w-2xl mx-auto">
                <h1 class="text-2xl font-semibold mb-2">Kepler Object of Interest (KOI) Predictor</h1>
                <p class="text-gray-400 mb-6">Enter a Kepler ID (e.g., K00752.01) to get the official classification and a prediction from our Random Forest model.</p>
                <form action="{{ url_for('home') }}" method="post">
                    <div class="flex flex-col sm:flex-row gap-4">
                        <input type="text" name="kepler_id" placeholder="Enter Kepler ID" class="flex-grow bg-gray-700 text-white border border-gray-600 rounded-md py-2 px-4 focus:outline-none focus:ring-2 focus:ring-blue-500" required>
                        <button type="submit" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-6 rounded-md transition duration-300">Analyze</button>
                    </div>
                </form>
                {{ result_html | safe }}
            </div>
            
            {% elif page == 'about' %}
            <div class="bg-gray-800 p-8 rounded-lg shadow-2xl max-w-4xl mx-auto text-gray-300">
                <h1 class="text-3xl font-bold mb-4 text-white">About Exogen</h1>
                <p class="mb-4">Exogen is a web application designed to predict the status of Kepler Objects of Interest (KOIs) using machine learning. It leverages data from the NASA Exoplanet Archive to train a Random Forest classifier, a powerful model capable of identifying complex patterns in data.</p>
                <h2 class="text-2xl font-semibold mb-2 text-white">The Mission</h2>
                <p class="mb-4">The goal of this project is to provide a simple, interactive tool for exploring the likelihood of a KOI being a confirmed exoplanet, a candidate, or a false positive. By inputting a Kepler ID, users can instantly see both the official NASA classification and a prediction generated by our custom-trained model.</p>
                <h2 class="text-2xl font-semibold mb-2 text-white">The Technology</h2>
                <ul class="list-disc list-inside space-y-2">
                    <li><strong>Backend:</strong> The application is powered by Python and the Flask micro web framework.</li>
                    <li><strong>Machine Learning:</strong> We use the scikit-learn library to build, train, and evaluate our Random Forest model. Missing data is handled using median imputation for robustness.</li>
                    <li><strong>Frontend:</strong> The user interface is built with standard HTML and styled with Tailwind CSS for a modern, responsive design.</li>
                </ul>
            </div>
            
            {% elif page == 'data' %}
            <div class="bg-gray-800 p-8 rounded-lg shadow-2xl max-w-6xl mx-auto text-gray-300">
                <h1 class="text-3xl font-bold mb-4 text-white">Model Performance & Insights</h1>
                <p class="mb-8">This page visualizes the performance of our trained Random Forest model. The model was trained on 80% of the preprocessed Kepler dataset and evaluated on the remaining 20%.</p>
                <div class="text-center mb-10">
                    <h2 class="text-lg font-semibold text-gray-400 uppercase">Model Accuracy</h2>
                    <p class="text-6xl font-bold text-green-400 mt-2">{{ accuracy_text }}</p>
                </div>
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div>
                        <h2 class="text-2xl font-semibold mb-4 text-white">Confusion Matrix</h2>
                        <p class="mb-4">This confusion matrix is a table displaying the performance in terms of amounts of true positives, false positives, true negatives, and false negatives.</p>
                        {% if cm_img %}
                            <img src="data:image/png;base64,{{ cm_img }}" alt="Confusion Matrix" class="rounded-lg shadow-lg">
                        {% else %}
                            <div class="p-4 rounded-md bg-yellow-900/50 border border-yellow-700 text-yellow-300"><strong>Image not available.</strong><br>Please delete the <code>exoplanet_model.joblib</code> file and restart the app to generate the charts.</div>
                        {% endif %}
                    </div>
                    <div>
                        <h2 class="text-2xl font-semibold mb-4 text-white">Feature Importance</h2>
                        <p class="mb-4">This chart is a ranking of the importance of each feature used when making predictions about an object's status.</p>
                        {% if fi_img %}
                            <img src="data:image/png;base64,{{ fi_img }}" alt="Feature Importance" class="rounded-lg shadow-lg">
                        {% else %}
                            <div class="p-4 rounded-md bg-yellow-900/50 border border-yellow-700 text-yellow-300"><strong>Image not available.</strong><br>Please delete the <code>exoplanet_model.joblib</code> file and restart the app to generate the charts.</div>
                        {% endif %}
                    </div>
                </div>
            </div>

            {% elif page == 'predict_csv' %}
            <div class="bg-gray-800 p-8 rounded-lg shadow-2xl max-w-2xl mx-auto">
                <h1 class="text-2xl font-semibold mb-2">Predict from CSV</h1>
                <p class="text-gray-400 mb-6">Upload a CSV file with the same features as the Kepler dataset to get a prediction for each row using the pre-trained model.</p>
                <form action="{{ url_for('predict_csv') }}" method="post" enctype="multipart/form-data">
                    <div class="flex flex-col gap-4">
                        <input type="file" name="file" class="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700" required>
                        <button type="submit" class="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-6 rounded-md transition duration-300">Upload and Predict</button>
                    </div>
                </form>
            </div>

            {% elif page == 'results' %}
            <div class="bg-gray-800 p-8 rounded-lg shadow-2xl max-w-7xl mx-auto">
                <h1 class="text-3xl font-bold mb-4 text-white">Prediction Results</h1>
                <p class="text-gray-400 mb-6">Showing predictions for the uploaded file using the Kepler-trained model.</p>
                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-gray-700">
                        <thead class="bg-gray-700">
                            <tr>
                                {% for header in headers %}
                                    <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">{{ header | replace('_', ' ') }}</th>
                                {% endfor %}
                            </tr>
                        </thead>
                        <tbody class="bg-gray-800 divide-y divide-gray-700">
                            {% for row in results %}
                                <tr>
                                    {% for header in headers %}
                                        <td class="px-6 py-4 whitespace-nowrap text-sm {% if header == 'Model Prediction' %}font-bold text-purple-400{% else %}text-gray-300{% endif %}">
                                            {{ row[header] }}
                                        </td>
                                    {% endfor %}
                                </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
            {% endif %}
        </main>
        <footer class="text-center text-gray-500 mt-12">
            <p>&copy; 2025 Exogen. Powered by Flask & scikit-learn.</p>
        </footer>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    result_html = ""
    if request.method == 'POST':
        kepler_id = request.form.get('kepler_id')
        target_obj = full_df[full_df['kepoi_name'] == kepler_id]

        if not target_obj.empty:
            model = artifacts['model']; imputer = artifacts['imputer']; model_features = artifacts['features']
            official_status = target_obj.iloc[0]['koi_disposition']
            features_for_model = target_obj[model_features]
            imputed_features = imputer.transform(features_for_model)
            prediction_numeric = model.predict(imputed_features)[0]
            prediction_label = DISPOSITION_MAP.get(prediction_numeric, "UNKNOWN")
            
            result_html = render_template_string("""
            <div class='mt-8 p-6 rounded-lg bg-gray-700/50 border border-gray-600'>
                <h2 class='text-xl font-semibold mb-4'>Analysis for <span class='font-mono text-blue-400'>{{kepler_id}}</span></h2>
                <div class='grid grid-cols-1 md:grid-cols-2 gap-4 text-center'>
                    <div class='bg-gray-800 p-4 rounded-md'><h3 class='text-sm font-bold text-gray-400'>OFFICIAL STATUS</h3><p class='text-2xl font-semibold text-green-400 mt-1'>{{official_status}}</p></div>
                    <div class='bg-gray-800 p-4 rounded-md'><h3 class='text-sm font-bold text-gray-400'>MODEL PREDICTION</h3><p class='text-2xl font-semibold text-purple-400 mt-1'>{{prediction_label}}</p></div>
                </div>
            </div>""", kepler_id=kepler_id, official_status=official_status, prediction_label=prediction_label)
        else:
            result_html = render_template_string("""
            <div class='mt-8 p-4 rounded-md bg-red-900/50 border border-red-700'>
                <h3 class='font-bold text-red-300'>Error</h3>
                <p class='text-red-400'>Kepler ID <span class='font-mono'>{{kepler_id}}</span> not found in the dataset.</p>
            </div>""", kepler_id=kepler_id)
            
    return render_template_string(HTML_TEMPLATE, page='home', title="Predictor", result_html=result_html)

@app.route('/about')
def about():
    return render_template_string(HTML_TEMPLATE, page='about', title="About")

@app.route('/data')
def data():
    acc = artifacts.get('accuracy', 0)
    cm_img = artifacts.get('confusion_matrix_img')
    fi_img = artifacts.get('feature_importance_img')
    
    return render_template_string(HTML_TEMPLATE, 
        page='data',
        title="Data & Insights",
        accuracy_text=f"{acc * 100:.2f}%" if acc else "N/A",
        cm_img=cm_img,
        fi_img=fi_img
    )

@app.route('/predict_csv', methods=['GET', 'POST'])
def predict_csv():
    if request.method == 'POST':
        # --- File Validation ---
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        if not file or not file.filename.endswith('.csv'):
            flash('Invalid file type. Please upload a .csv file.', 'error')
            return redirect(request.url)

        try:
            upload_df = pd.read_csv(file)
        except Exception as e:
            flash(f"Error reading CSV file: {e}", 'error')
            return redirect(request.url)

        # --- Feature Validation ---
        model_features = artifacts['features']
        # Find which columns are missing
        missing_cols = [col for col in model_features if col not in upload_df.columns]
        
        if missing_cols:
            # Create a more helpful error message
            error_message = "The uploaded CSV is missing required columns: <br><ul class='list-disc list-inside mt-2'>"
            for col in missing_cols:
                error_message += f"<li>{col}</li>"
            error_message += "</ul>"
            flash(error_message, 'error')
            return redirect(request.url)

        # --- Prediction Logic ---
        features_to_predict = upload_df[model_features]
        
        imputer = artifacts['imputer']
        model = artifacts['model']

        imputed_features = imputer.transform(features_to_predict)
        predictions_numeric = model.predict(imputed_features)
        predictions_labels = [DISPOSITION_MAP.get(p, "UNKNOWN") for p in predictions_numeric]

        upload_df['Model Prediction'] = predictions_labels
        
        results_list = upload_df.to_dict(orient='records')
        headers = upload_df.columns.tolist()

        return render_template_string(HTML_TEMPLATE, 
            page='results', 
            title='Prediction Results',
            results=results_list,
            headers=headers)

    # --- Show upload form on GET request ---
    return render_template_string(HTML_TEMPLATE, page='predict_csv', title='Predict from CSV')

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Open your web browser and go to: http://127.0.0.1:5000")
    app.run(debug=True)
