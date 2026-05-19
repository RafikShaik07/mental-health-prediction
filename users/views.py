from django.shortcuts import render
from django.conf import settings
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ML Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ===============================
# Training View
# ===============================
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from django.conf import settings
from django.shortcuts import render, redirect

# Sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ============================
# Load & preprocess dataset
# ============================
def load_preprocess_data():
    df = pd.read_csv(os.path.join(settings.BASE_DIR, 'media', 'Deepression_extended.csv'))
    df.dropna(inplace=True)
    
    X = df.drop(columns=['Depression State'])
    y = df['Depression State']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler, label_encoder

# ============================
# Utility to train & save classical ML model
# ============================
def train_and_save_classical(model, name, X_train, y_train, X_test, y_test, scaler, feature_cols, label_encoder):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=range(len(label_encoder.classes_)),
                                   target_names=label_encoder.classes_, zero_division=0, output_dict=True)
    
    joblib.dump({"model": model, "features": feature_cols.tolist(), "scaler": scaler, "label_encoder": label_encoder},
                os.path.join(settings.BASE_DIR, f'media/depression_{name.lower()}.pkl'))
    
    # Save a simple plot
    plt.figure(figsize=(4,4))
    plt.bar(['Accuracy'], [acc], color='skyblue')
    plt.ylim(0,1)
    plt.title(f"{name} Accuracy")
    plt.tight_layout()
    plot_path = os.path.join(settings.BASE_DIR, f'media/{name.lower()}_accuracy.png')
    plt.savefig(plot_path)
    plt.close()
    
    return acc, report, f'/media/{name.lower()}_accuracy.png'

# ============================
# Individual Algorithm Views
# ============================

def train_logistic_regression(request):
    X_train, X_test, y_train, y_test, feature_cols, scaler, label_encoder = load_preprocess_data()
    acc, report, graph_url = train_and_save_classical(LogisticRegression(max_iter=1000), "LogisticRegression",
                                                      X_train, y_train, X_test, y_test, scaler, feature_cols, label_encoder)
    return render(request, 'users/accuracy.html', {
    'algorithm': 'Logistic Regression',
    'accuracy': round(acc * 100, 2),
    'report': report,
    'graph_url': graph_url
})



def train_random_forest(request):
    X_train, X_test, y_train, y_test, feature_cols, scaler, label_encoder = load_preprocess_data()
    acc, report, graph_url = train_and_save_classical(RandomForestClassifier(n_estimators=100, random_state=42),
                                                      "RandomForest",
                                                      X_train, y_train, X_test, y_test, scaler, feature_cols, label_encoder)
    return render(request, 'users/accuracy.html', {
        'algorithm': 'Random Forest',
        'accuracy': round(acc, 4),
        'report': report,
        'graph_url': graph_url
    })


def train_knn(request):
    X_train, X_test, y_train, y_test, feature_cols, scaler, label_encoder = load_preprocess_data()
    acc, report, graph_url = train_and_save_classical(KNeighborsClassifier(), "KNN",
                                                      X_train, y_train, X_test, y_test, scaler, feature_cols, label_encoder)
    return render(request, 'users/accuracy.html', {
        'algorithm': 'K-Nearest Neighbors',
        'accuracy': round(acc, 4),
        'report': report,
        'graph_url': graph_url
    })


def train_svm(request):
    X_train, X_test, y_train, y_test, feature_cols, scaler, label_encoder = load_preprocess_data()
    acc, report, graph_url = train_and_save_classical(SVC(kernel='linear', probability=True), "SVM",
                                                      X_train, y_train, X_test, y_test, scaler, feature_cols, label_encoder)
    return render(request, 'users/accuracy.html', {
        'algorithm': 'Support Vector Machine',
        'accuracy': round(acc, 4),
        'report': report,
        'graph_url': graph_url
    })


import matplotlib.pyplot as plt
import numpy as np
import os
from django.conf import settings

def compare_models(request):
    X_train, X_test, y_train, y_test, feature_cols, scaler, label_encoder = load_preprocess_data()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(kernel='linear', probability=True),
    }

    accuracies = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies[name] = accuracy_score(y_test, y_pred)

    # Generate comparison graph
    model_names = list(accuracies.keys())
    accuracy_values = [v * 100 for v in accuracies.values()]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(model_names, accuracy_values)
    plt.xlabel("Models")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0, 100)

    # bar labels
    for bar, acc in zip(bars, accuracy_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{acc:.2f}%", ha='center')

    graph_path = os.path.join(settings.MEDIA_ROOT, "comparison_graph.png")
    plt.savefig(graph_path)
    plt.close()

    graph_url = settings.MEDIA_URL + "comparison_graph.png"

    return render(request, 'admins/graph.html', {"graph_url": graph_url})

# ===============================
# Prediction View
# ===============================
from django.shortcuts import render
from django.conf import settings
import os
import joblib
import pandas as pd
import numpy as np

# ML Imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ===============================
# Prediction View
# ===============================
def prediction(request):
    prediction_value = None
    error_message = None

    # Path to saved model
    model_path = os.path.join(settings.BASE_DIR, 'media', 'depression_randomforest.pkl')

    if not os.path.exists(model_path):
        error_message = "Model not trained yet. Please train the model first."
    else:
        try:
            model_dict = joblib.load(model_path)
            model = model_dict["model"]
            features = model_dict["features"]
            scaler = model_dict.get("scaler", None)
            label_encoder = model_dict.get("label_encoder", None)
        except Exception:
            error_message = "Saved model is not in correct format. Please retrain."

        if request.method == "POST" and not error_message:
            try:
                # Collect form inputs
                input_data = pd.DataFrame([{
                    "Sleep": float(request.POST.get("Sleep", 0)),
                    "Appetite": float(request.POST.get("Appetite", 0)),
                    "Interest": float(request.POST.get("Interest", 0)),
                    "Fatigue": float(request.POST.get("Fatigue", 0)),
                    "Worthlessness": float(request.POST.get("Worthlessness", 0)),
                    "Concentration": float(request.POST.get("Concentration", 0)),
                    "Agitation": float(request.POST.get("Agitation", 0)),
                    "Suicidal Ideation": float(request.POST.get("Suicidal_Ideation", 0)),
                    "Sleep Disturbance": float(request.POST.get("Sleep_Disturbance", 0)),
                    
                    "Panic Attacks": float(request.POST.get("Panic_Attacks", 0)),
              
                    "Restlessness": float(request.POST.get("Restlessness", 0)),
                   
                }])

                # Align columns to training features
                for col in features:
                    if col not in input_data.columns:
                        input_data[col] = 0
                input_data = input_data[features]

                # Scale inputs
                if scaler:
                    input_data = scaler.transform(input_data)

                # Predict class
                pred_class = model.predict(input_data)[0]

                # Map numeric class to label
                if label_encoder:
                    prediction_value = label_encoder.inverse_transform([pred_class])[0]
                else:
                    # fallback mapping
                    mapping = {0: "No Depression", 1: "Mild", 2: "Moderate", 3: "Severe"}
                    prediction_value = mapping.get(pred_class, "Unknown")

            except Exception as e:
                error_message = f"Error in prediction: {str(e)}"

    return render(request, "users/prediction.html", {
        "prediction": prediction_value,
        "error": error_message
    })



# ===============================
# View Dataset
# ===============================
def ViewDataset(request):
    dataset_path = os.path.join(settings.MEDIA_ROOT, 'Deepression_extended.csv')
    df = pd.read_csv(dataset_path, nrows=100)
    df_html = df.to_html(index=False)
    return render(request, 'users/viewData.html', {'data': df_html})


# ===============================
# User Registration
# ===============================
from django.contrib import messages
from .models import UserRegistrationModel

def UserRegisterActions(request):
    if request.method == 'POST':
        user = UserRegistrationModel(
            name=request.POST['name'],
            loginid=request.POST['loginid'],
            password=request.POST['password'],
            email=request.POST['email'],
            address=request.POST['address'],
            status='waiting'
        )
        user.save()
        messages.success(request,"Registration successful!")
    return render(request, 'UserRegistrations.html')


# ===============================
# User Login
# ===============================
def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                return render(request, 'users/UserHomePage.html')
            else:
                messages.success(request, 'Your account is not activated yet.')
        except UserRegistrationModel.DoesNotExist:
            messages.success(request, 'Invalid Login ID or Password.')
    return render(request, 'UserLogin.html')


# ===============================
# User Home & Index
# ===============================
def UserHome(request):
    return render(request, 'users/UserHomePage.html')


def index(request):
    return render(request,"index.html")
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect
from django.conf import settings
import os

def upload_dataset_for_algo(request, algorithm):
    """
    Uploads dataset, saves it, and redirects to the respective training function.
    """
    if request.method == 'POST' and request.FILES.get('dataset'):
        dataset = request.FILES['dataset']
        fs = FileSystemStorage(location=os.path.join(settings.BASE_DIR, 'media'))
        fs.save('Deepression_extended.csv', dataset)

        # Redirect to respective model after upload
        if algorithm == 'LR':
            return redirect('train_logistic_regression')
        elif algorithm == 'RF':
            return redirect('train_random_forest')
        elif algorithm == 'KNN':
            return redirect('train_knn')
        elif algorithm == 'SVM':
            return redirect('train_svm')

    return render(request, 'admins/upload_dataset_algo.html', {'algorithm': algorithm})

