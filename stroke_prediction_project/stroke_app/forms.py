from django import forms

class StrokePredictionForm(forms.Form):
    age = forms.IntegerField(min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    hypertension = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')], widget=forms.Select(attrs={'class': 'form-control'}))
    heart_disease = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')], widget=forms.Select(attrs={'class': 'form-control'}))
    avg_glucose_level = forms.FloatField(min_value=50, max_value=250, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    bmi = forms.FloatField(min_value=10, max_value=40, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    gender = forms.ChoiceField(choices=[('Male', 'Male'), ('Female', 'Female')], widget=forms.Select(attrs={'class': 'form-control'}))
    ever_married = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')], widget=forms.Select(attrs={'class': 'form-control'}))
    work_type = forms.ChoiceField(choices=[('Private', 'Private'), ('Self-employed', 'Self-employed'), ('Govt_job', 'Government Job'), ('children', 'Children'), ('Never_worked', 'Never Worked')], widget=forms.Select(attrs={'class': 'form-control'}))
    residence_type = forms.ChoiceField(choices=[('Urban', 'Urban'), ('Rural', 'Rural')], widget=forms.Select(attrs={'class': 'form-control'}))
    smoking_status = forms.ChoiceField(choices=[('formerly smoked', 'Formerly Smoked'), ('never smoked', 'Never Smoked'), ('smokes', 'Smokes'), ('Unknown', 'Unknown')], widget=forms.Select(attrs={'class': 'form-control'}))