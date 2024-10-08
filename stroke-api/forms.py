# prediction/forms.py
from django import forms

class StrokeRiskForm(forms.Form):
    age = forms.IntegerField(label='Age', min_value=0, max_value=100)
    hypertension = forms.ChoiceField(label='Hypertension', choices=[('Yes', 'Yes'), ('No', 'No')])
    heart_disease = forms.ChoiceField(label='Heart Disease', choices=[('Yes', 'Yes'), ('No', 'No')])
    avg_glucose_level = forms.FloatField(label='Average Glucose Level', min_value=50, max_value=250)
    bmi = forms.FloatField(label='BMI', min_value=10, max_value=40)
    gender = forms.ChoiceField(label='Gender', choices=[('Male', 'Male'), ('Female', 'Female')])
    ever_married = forms.ChoiceField(label='Ever Married', choices=[('Yes', 'Yes'), ('No', 'No')])
    work_type = forms.ChoiceField(label='Work Type', choices=[('Private', 'Private'), ('Self-employed', 'Self-employed'), ('Govt_job', 'Govt_job'), ('children', 'children'), ('Never_worked', 'Never_worked')])
    residence_type = forms.ChoiceField(label='Residence Type', choices=[('Urban', 'Urban'), ('Rural', 'Rural')])
    smoking_status = forms.ChoiceField(label='Smoking Status', choices=[('formerly smoked', 'formerly smoked'), ('never smoked', 'never smoked'), ('smokes', 'smokes'), ('Unknown', 'Unknown')])
