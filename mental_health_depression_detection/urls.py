"""
URL configuration for cyberattack_detection project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from mental_health_depression_detection import views as mainView
from admins import views as admins
from users import views as usr
from django.contrib.staticfiles.urls import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.conf import settings
from users import views as uv
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", mainView.index, name="index"),
    path("index", mainView.index, name="index"),
    # path("Adminlogin", mainView.AdminLogin, name="AdminLogin"),
    path("UserLogin", mainView.UserLogin, name="UserLogin"),

    # admin views
    path("AdminLogincheck", admins.AdminLoginCheck, name="AdminLoginCheck"),
    path('userDetails', admins.RegisterUsersView, name='RegisterUsersView'),
    path('ActivUsers/', admins.ActivaUsers, name='activate_users'),
    path('DeleteUsers/', admins.DeleteUsers, name='delete_users'),
    path('adminhome',admins.adminhome,name="adminhome"),

   
    path('upload_dataset/<str:algorithm>/', usr.upload_dataset_for_algo, name='upload_dataset_for_algo'),
    path('train_logistic_regression/', usr.train_logistic_regression, name='train_logistic_regression'),
    path('train_random_forest/', usr.train_random_forest, name='train_random_forest'),
    path('train_knn/', usr.train_knn, name='train_knn'),
    path('train_svm/', usr.train_svm, name='train_svm'),


    
    
    #userurls
    path('UserRegisterForm',uv.UserRegisterActions,name='UserRegisterForm'),
    path("UserLoginCheck/", usr.UserLoginCheck, name="UserLoginCheck"),
    path("UserHome/", usr.UserHome, name="UserHome"),
    path("ViewDataset/", usr.ViewDataset, name="ViewDataset"),
    path("prediction/", usr.prediction, name="prediction"),
    path("train_logistic_regression", usr.train_logistic_regression, name="train_logistic_regression"),
    path("train_random_forest", usr.train_random_forest, name="train_random_forest"),
    path("train_knn", usr.train_knn, name="train_knn"),
    path("train_svm", usr.train_svm, name="train_svm"),
    path("compare_models", usr.compare_models, name="compare_models"),
    path("index/", usr.index, name="index"),
   



]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


