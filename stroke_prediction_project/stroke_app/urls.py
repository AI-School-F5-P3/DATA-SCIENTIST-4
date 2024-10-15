from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_stroke, name='predict_stroke'),
    path('upload/', views.upload_view, name='upload'),
]