from django.urls import path
from . import views

urlpatterns = [
    path('run', views.index, name='index'),
    path('test_files', views.get_test_file_list, name='get_test_file_list'),
    path('test_file/upload', views.upload_java_test_file, name='upload_java_test_file'),
    path('test_file/delete', views.delete_test, name='delete_test_file'),
    path('generate/grade', views.generate_grade, name='generate_grade'),
    
    path('dynamic_compile/', views.dynamic_compile, name='dynamic_compile'),
    path('dynamic_compile_llm/', views.dynamic_compile_llm, name='dynamic_compile_llm'),
    path('dynamic_test/', views.dynamic_test, name='dynamic_test'),
    path('dynamic_test_with_database_junit/', views.dynamic_test_with_database_junit, name='dynamic_test_with_database_junit'),
    
    path('generate-question-with-junit/', views.generate_question_with_junit, name='generate_question_with_junit'),
    path('save-junit-test-file/', views.save_junit_test_file, name='save_junit_test_file'),
    
    path('history/', views.get_history_list, name='get_history_list'),
    path('history/<int:history_id>/', views.get_history_detail, name='get_history_detail'),
    path('content/<int:content_id>/history/', views.get_content_history, name='get_content_history'),
    path('history/search/', views.search_history, name='search_history'),
    
    path('evaluate-rag-retrieval/', views.evaluate_rag_retrieval, name='evaluate_rag_retrieval'),
    path('run-rag-evaluation-pipeline/', views.run_rag_evaluation_pipeline, name='run_rag_evaluation_pipeline'),
]