from django.shortcuts import render

def index(request):
    return render(request, "cutting_app/index.html")
