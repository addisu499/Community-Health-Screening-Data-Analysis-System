
# Assignment Title: Community Health Screening Data Analysis System


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# TASK 1: CREATE THE DATASET
# ==========================================
print("--- TASK 1: CREATING DATASET ---")

# Initial dictionary containing 11 participant records (including a duplicate and missing values)
raw_data = {
    'Patient_ID': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P009', 'P010', 'P010'],
    'Age': [28, 45, 62, 35, 50, 23, 41, 58, 31, 65, 65],
    'Gender': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Male'],
    'Height_cm': [165, 180, 155, 175, 160, 185, 168, 172, 162, 178, 178],
    'Weight_kg': [55.0, 85.0, np.nan, 70.0, 95.0, 62.0, 78.0, 88.0, 52.0, 90.0, 90.0], # np.nan is the first missing value
    'Systolic_BP': [115, 135, 142, 122, 148, 118, 128, 130, 112, 145, 145],
    'Diastolic_BP': [75, 85, 92, 78, 95, 76, 82, 84, 70, 88, 88],
    'Blood_Sugar': [90, 105, 130, 95, 145, 85, 110, 115, 88, 120, 120],

    'Physical_Activity': [4.0, 2.5, 1.0, 5.0, 0.0, 6.0, 3.0, 1.5, 4.5, 2.0, 2.0],
    'Smoking_Status': ['No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', None, 'Yes', 'Yes'] # None is the second missing value
}

# Convert the dictionary to a Pandas DataFrame
df = pd.DataFrame(raw_data)

# Display the first five records
print("First five records of the raw DataFrame:")
print(df.head())
print("\n")


# ==========================================
# TASK 2: DATA CLEANING
# ==========================================
print("--- TASK 2: DATA CLEANING ---")
\\
# 1. Display total duplicates and missing values before cleaning
print(f"Total rows before removing duplicates: {len(df)}")
print("Missing values per column before cleaning:")
print(df.isnull().sum())

# 2. Remove duplicate rows
df = df.drop_duplicates()
print(f"Total rows after removing duplicates: {len(df)}")

# 3. Replace missing numerical value in 'Weight_kg' with the median
median_weight = df['Weight_kg'].median()
df['Weight_kg'] = df['Weight_kg'].fillna(median_weight)
print(f"Filled missing weight with median: {median_weight} kg")

# 4. Replace missing categorical value in 'Smoking_Status' with the most frequent value (mode)
most_frequent_smoke = df['Smoking_Status'].mode()[0]
df['Smoking_Status'] = df['Smoking_Status'].fillna(most_frequent_smoke)
print(f"Filled missing smoking status with mode: '{most_frequent_smoke}'")

# 5. Verify no missing values remain
print("\nMissing values remaining in the cleaned DataFrame:")
print(df.isnull().sum())
print("\n")


# ==========================================
# TASK 3: CREATE HEALTH INDICATORS
# ==========================================
print("--- TASK 3: CREATING HEALTH INDICATORS ---")

# User-defined function to calculate BMI
def calculate_bmi(weight, height_cm):
    height_m = height_cm / 100.0
    return round(weight / (height_m ** 2), 1)

# Apply BMI calculation to the DataFrame
df['BMI'] = df.apply(lambda row: calculate_bmi(row['Weight_kg'], row['Height_cm']), axis=1)


# User-defined function to categorize BMI
def get_bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25.0:
        return 'Normal'
    elif 25.0 <= bmi < 30.0:
        return 'Overweight'
    else:
        return 'Obese'

# Apply BMI categorization
df['BMI_Category'] = df['BMI'].apply(get_bmi_category)


# User-defined function to categorize Blood Pressure
def get_bp_category(systolic, diastolic):
    if systolic < 120 and diastolic < 80:
        return 'Normal'
    elif 120 <= systolic < 130 and diastolic < 80:
        return 'Elevated'
    elif (130 <= systolic < 140) or (80 <= diastolic < 90):
        return 'High Stage 1'
    else:  # systolic >= 140 or diastolic >= 90
        return 'High Stage 2'

# Apply Blood Pressure categorization
df['BP_Category'] = df.apply(lambda row: get_bp_category(row['Systolic_BP'], row['Diastolic_BP']), axis=1)


# User-defined function to assess overall health risk
def get_health_risk(bmi, bp_category, blood_sugar, smoking):
    # High Risk conditions
    if bmi >= 30.0 or bp_category == 'High Stage 2' or blood_sugar >= 126 or (smoking == 'Yes' and bp_category != 'Normal'):
        return 'High Risk'
    # Moderate Risk conditions
    elif (25.0 <= bmi < 30.0) or bp_category in ['Elevated', 'High Stage 1'] or (100 <= blood_sugar < 126):
        return 'Moderate Risk'
    # Low Risk
    else:
        return 'Low Risk'

# Apply Overall Health Risk categorization
df['Overall_Health_Risk'] = df.apply(lambda row: get_health_risk(
    row['BMI'], row['BP_Category'], row['Blood_Sugar'], row['Smoking_Status']
), axis=1)

print("DataFrame with newly added health indicators:")
print(df[['Patient_ID', 'BMI', 'BMI_Category', 'BP_Category', 'Overall_Health_Risk']])
print("\n")


# ==========================================
# TASK 4: NUMPY ANALYSIS
# ==========================================
print("--- TASK 4: NUMPY ANALYSIS ---")

# Convert pandas columns to NumPy arrays
bmi_array = df['BMI'].to_numpy()
blood_sugar_array = df['Blood_Sugar'].to_numpy()

# Calculate statistics for BMI
print("BMI Statistics:")
print(f"  Mean: {np.mean(bmi_array):.2f}")
print(f"  Min:  {np.min(bmi_array):.1f}")
print(f"  Max:  {np.max(bmi_array):.1f}")
print(f"  Std:  {np.std(bmi_array):.2f}")

# Calculate statistics for Blood Sugar
print("Blood Sugar Statistics (mg/dL):")
print(f"  Mean: {np.mean(blood_sugar_array):.2f}")
print(f"  Min:  {np.min(blood_sugar_array)}")
print(f"  Max:  {np.max(blood_sugar_array)}")
print(f"  Std:  {np.std(blood_sugar_array):.2f}")
print("\n")


# ==========================================
# TASK 5: DATA ANALYSIS USING PANDAS
# ==========================================
print("--- TASK 5: PANDAS ANALYSIS ---")

# 1. Average BMI and Blood Sugar by Gender
print("Average BMI and Blood Sugar by Gender:")
avg_by_gender = df.groupby('Gender')[['BMI', 'Blood_Sugar']].mean()
print(avg_by_gender)
print("\n")

# 2. Number of patients by BMI category and Health Risk category
print("Patient Count by BMI Category:")
print(df['BMI_Category'].value_counts())
print("\n")

print("Patient Count by Overall Health Risk:")
print(df['Overall_Health_Risk'].value_counts())
print("\n")

# 3. Average age by Gender
print("Average Age by Gender:")
avg_age_by_gender = df.groupby('Gender')['Age'].mean()
print(avg_age_by_gender)
print("\n")

# 4. Percentage of Smokers
smoker_count = len(df[df['Smoking_Status'] == 'Yes'])
total_count = len(df)
smoker_percentage = (smoker_count / total_count) * 100
print(f"Percentage of Smokers in the campaign: {smoker_percentage:.1f}%")
print("\n")


# ==========================================
# TASK 6: DATA VISUALIZATION
# ==========================================
print("--- TASK 6: GENERATING PLOTS ---")

# Set seaborn plotting style for cleaner visuals
sns.set_theme(style="whitegrid")

# Create a 2x2 grid of subplots for all 4 charts
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: BMI Histogram
sns.histplot(df['BMI'], bins=5, kde=True, color='teal', ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Body Mass Index (BMI)')
axes[0, 0].set_xlabel('BMI')
axes[0, 0].set_ylabel('Frequency')

# Plot 2: Health Risk Category Bar Chart
sns.countplot(x='Overall_Health_Risk', data=df, hue='Overall_Health_Risk', palette='Oranges_r', legend=False, ax=axes[0, 1])
axes[0, 1].set_title('Number of Patients by Health Risk Level')
axes[0, 1].set_xlabel('Health Risk Category')
axes[0, 1].set_ylabel('Count of Patients')

# Plot 3: Age vs Blood Sugar Level Scatter Plot
sns.scatterplot(x='Age', y='Blood_Sugar', data=df, hue='Gender', style='Smoking_Status', s=100, ax=axes[1, 0])
axes[1, 0].set_title('Age vs. Blood Sugar Level')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Blood Sugar Level (mg/dL)')

# Plot 4: Correlation Heatmap
# Select only numerical columns for correlation matrix calculation
numerical_cols = ['Age', 'Height_cm', 'Weight_kg', 'Systolic_BP', 'Diastolic_BP', 'Blood_Sugar', 'Physical_Activity', 'BMI']
corr_matrix = df[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=axes[1, 1])
axes[1, 1].set_title('Correlation Heatmap of Numerical Features')

# Adjust layout automatically so everything fits beautifully without overlap
plt.tight_layout()

# Display the charts
plt.show()