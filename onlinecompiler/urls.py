"""onlinecompiler URL Configuration"""
from django.contrib import admin
from django.urls import path, include
from compiler import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('compiler/', include('compiler.urls')),
    path('', views.health_check, name='health_check'),
    path('api/', include('compiler.urls')),
]
