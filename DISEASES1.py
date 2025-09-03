import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define the base path where your CSV files are located
BASE_PATH = "C:/Users/nanda/OneDrive/Pictures/PROJECT2/"

# Load datasets
try:
    diabetes_data = pd.read_csv(f"{BASE_PATH}diabetes.csv")
    heart_data = pd.read_csv(f"{BASE_PATH}heart.csv")
    parkinsons_data = pd.read_csv(f"{BASE_PATH}parkinsons.csv")
except FileNotFoundError as e:
    messagebox.showerror("File Error", f"Could not find the data file: {e}\nPlease ensure the files are in the specified path.")
    exit()

# --- Diabetes Model ---
X_d = diabetes_data.drop('Outcome', axis=1)
y_d = diabetes_data['Outcome']
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.2, random_state=42)
model_d = RandomForestClassifier().fit(X_train_d, y_train_d)

# --- Heart Disease Model ---
# Separate features (X_h) and target (y_h)
X_h = heart_data.drop('HeartDisease', axis=1)
y_h = heart_data['HeartDisease']

# Identify categorical columns for one-hot encoding
categorical_features_h = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Create a column transformer to apply OneHotEncoder to categorical features
preprocessor_h = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_h)
    ],
    remainder='passthrough' # Keep other numerical columns as they are
)

# Create a pipeline that first preprocesses the data and then trains the RandomForestClassifier
model_h_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_h),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit the pipeline model for heart disease
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_h, y_h, test_size=0.2, random_state=42)
model_h_pipeline.fit(X_train_h, y_train_h)

# --- Parkinson's Model ---
X_p = parkinsons_data.drop(['name', 'status'], axis=1)
y_p = parkinsons_data['status']
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_p, y_p, test_size=0.2, random_state=42)
model_p = RandomForestClassifier().fit(X_train_p, y_train_p)


# --- GUI Application Setup ---
root = tk.Tk()
root.title("Disease Prediction System")
root.geometry("600x600")

disease_var = tk.StringVar()
disease_choices = ["Diabetes", "Heart Disease", "Parkinson’s"]
ttk.Label(root, text="Select Disease:", font=('Arial', 14)).pack(pady=10)
disease_combo = ttk.Combobox(root, values=disease_choices, textvariable=disease_var, state='readonly', font=('Arial', 12))
disease_combo.pack()

frame = tk.Frame(root)
frame.pack(pady=20)

entries = [] # To store the input entry widgets
current_labels = [] # To store the original labels for the current disease form

def clear_form():
    """Clears the current input form fields."""
    for widget in frame.winfo_children():
        widget.destroy()
    entries.clear()
    current_labels.clear() # Clear labels too

def predict():
    """Performs prediction based on the selected disease and user inputs."""
    disease = disease_var.get()
    if not disease:
        messagebox.showerror("Selection Error", "Please select a disease first.")
        return

    input_values = [entry.get() for entry in entries]
    input_dict = dict(zip(current_labels, input_values)) # Create a dictionary from labels and values

    try:
        # Convert numerical fields to float, keeping categorical as strings for preprocessing
        processed_input = {}
        for label, value in input_dict.items():
            if disease == "Diabetes" or disease == "Parkinson’s": # These models expect all numerical input
                processed_input[label] = float(value)
            elif disease == "Heart Disease": # Handle categorical features for Heart Disease
                if label in categorical_features_h:
                    processed_input[label] = value # Keep as string for one-hot encoding
                else:
                    processed_input[label] = float(value)
        
        # Convert the processed_input dictionary to a DataFrame for prediction
        input_df = pd.DataFrame([processed_input])

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values for all numeric fields.")
        return

    # Perform prediction based on the selected disease
    if disease == "Diabetes":
        result = model_d.predict(input_df)[0]
        messagebox.showinfo("Prediction Result", "Diabetes: Positive" if result == 1 else "Diabetes: Negative")
    elif disease == "Heart Disease":
        # Use the pipeline for prediction, it handles preprocessing
        result = model_h_pipeline.predict(input_df)[0]
        messagebox.showinfo("Prediction Result", "Heart Disease: Present" if result == 1 else "Heart Disease: Absent")
    elif disease == "Parkinson’s":
        result = model_p.predict(input_df)[0]
        messagebox.showinfo("Prediction Result", "Parkinson’s: Positive" if result == 1 else "Parkinson’s: Negative")


def show_form(event=None):
    """Dynamically creates input fields based on the selected disease."""
    clear_form() # Clear any existing fields
    disease = disease_var.get()

    labels_to_display = []
    if disease == "Diabetes":
        labels_to_display = X_d.columns.tolist()
    elif disease == "Heart Disease":
        labels_to_display = X_h.columns.tolist()
    elif disease == "Parkinson’s":
        labels_to_display = X_p.columns.tolist()
    else:
        return # Do nothing if no disease is selected

    current_labels.extend(labels_to_display) # Store labels for prediction function

    # Create a label and entry for each feature
    for label_text in labels_to_display:
        row = tk.Frame(frame)
        row.pack(fill='x', pady=2)
        tk.Label(row, text=label_text + ":", width=25, anchor='w', font=('Arial', 10)).pack(side='left')
        ent = tk.Entry(row, font=('Arial', 10))
        ent.pack(side='right', expand=True, fill='x')
        entries.append(ent)

    # Add a Predict button at the bottom of the form
    # Destroy and recreate to ensure it's always at the bottom and functional
    for widget in root.winfo_children():
        if isinstance(widget, tk.Button) and widget.cget("text") == "Predict":
            widget.destroy()
    tk.Button(root, text="Predict", command=predict, bg='green', fg='white', font=('Arial', 12, 'bold')).pack(pady=15)


# Bind the combobox selection event to the show_form function
disease_combo.bind("<<ComboboxSelected>>", show_form)

# Run the Tkinter event loop
root.mainloop()