from django.urls import path
from . import views
from .views import upload_csv_and_predict

urlpatterns = [
    path('', views.predict_stroke, name='predict_stroke'),
    path('upload/', upload_csv_and_predict, name='upload'),
]