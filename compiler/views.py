import os
from django.http import JsonResponse

from lib.FileUploader import FileUploader
from lib.java_runner import JavaRunner
from django.views.decorators.csrf import csrf_exempt

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
import torch
import re

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import nltk
import numpy as np

# Import RAG evaluation components
from rag.rag_service import RAGService

@csrf_exempt
def index(request):
    print("="*50)
    print("INDEX ENDPOINT CALLED!")
    print("="*50)
    
    user_dir = request.POST["user"]
    code = request.POST["code"]
    question_id = request.POST.get("question_id")
    
    # Check if question_data is provided for strict validation
    question_data_raw = request.POST.get("question_data")
    question_data = None
    
    print(f"POST data keys: {list(request.POST.keys())}")
    print(f"question_id: {question_id}")
    print(f"question_data_raw present: {question_data_raw is not None}")
    
    if question_data_raw:
        try:
            import json
            question_data = json.loads(question_data_raw)
            print(f"✅ PARSED question_data successfully: {len(question_data)} keys")
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing question_data in index: {e}")
    
    # Create JavaRunner with question_data for validation
    java_runner = JavaRunner(user_directory=user_dir, code=code, question_data=question_data)
    
    # Set question_id for file-based JUnit loading
    if question_id:
        java_runner.question_id = question_id
    
    # Use strict validation if question_data is provided, or try file-based tests
    if question_data:
        print("🔥 CALLING run_with_validation()")
        res = java_runner.run_with_validation()
    else:
        print("🔥 CALLING standard run() with potential file-based JUnit")
        res = java_runner.run()

    print(f"Result keys: {list(res.keys())}")
    print("="*50)

    return JsonResponse({
        'output': res
    }, status=200)

@csrf_exempt
def health_check(request):
    """Health check endpoint"""
    return JsonResponse({"status": "OK", "service": "Django LLM Compiler"})


@csrf_exempt
def upload_java_test_file(request):
    file = request.FILES["file"]
    fu = FileUploader(filename=file, file=file)
    fu.upload()

    return JsonResponse({
        "status": "ok"
    }, status=200)


def get_test_file_list(request):
    print(request)
    return JsonResponse({
        "file_list": os.listdir("java_files/test_cases")
    }, status=200)


@csrf_exempt
def delete_test(request):
    filename = request.POST["filename"]
    print(filename)
    try:
        os.remove("java_files/test_cases/{}".format(filename))
    except FileNotFoundError as err:
        return JsonResponse({"error": str(err), "status": "failed"}, status=500)

    return JsonResponse({"message": "{} deleted".format(filename), "status": "success"}, status=200)


@csrf_exempt
def generate_grade(request):
  
    esay_answer = request.POST["esay_answer"]
    esay_answer2 = request.POST["esay_answer2"]
    esay_answer3 = request.POST["esay_answer3"]
    esay_answer4 = request.POST["esay_answer4"]
    user_answer = request.POST["user_answer"]

    # Fungsi untuk mendapatkan sinonim dari sebuah kata menggunakan WordNet Bahasa Indonesia
    # def get_synonyms(word):
    #     synonyms = []
    #     for synset in wordnet.synsets(word, lang='ind'):
    #         for lemma in synset.lemma_names('ind'):
    #             synonyms.append(lemma)
    #     return set(synonyms)

    # Memecah kalimat menjadi kata-kata
    # kata_kata1 = nltk.word_tokenize(user_answer)
    # kata_kata2 = nltk.word_tokenize(esay_answer)
    # kata_kata3 = nltk.word_tokenize(esay_answer2)
    # kata_kata4 = nltk.word_tokenize(esay_answer3)
    # kata_kata5 = nltk.word_tokenize(esay_answer4)

    # Mencari sinonim dari setiap kata dalam kalimat2 dan mengganti jika sinonimnya ada dalam kalimat1
    # kalimat1_dengan_kalimat2 = []
    # for kata in kata_kata1:
    #     sinonim_kata1 = get_synonyms(kata)
    #     if kata in kata_kata2:
    #         kalimat1_dengan_kalimat2.append(kata)
    #     elif sinonim_kata1.intersection(set(kata_kata2)):
    #         kalimat1_dengan_kalimat2.append(list(sinonim_kata1.intersection(set(kata_kata2)))[0])
    #     else:
    #         kalimat1_dengan_kalimat2.append(kata)

    # user_answer = ' '.join(kalimat1_dengan_kalimat2)
    # print("\nKalimat 1 dengan sinonim yang berasal dari kalimat 2:")
    # print(user_answer)

    # kalimat1_dengan_kalimat3 = []
    # for kata in kata_kata1:
    #     sinonim_kata1 = get_synonyms(kata)
    #     if kata in kata_kata3:
    #         kalimat1_dengan_kalimat3.append(kata)
    #     elif sinonim_kata1.intersection(set(kata_kata3)):
    #         kalimat1_dengan_kalimat3.append(list(sinonim_kata1.intersection(set(kata_kata3)))[0])
    #     else:
    #         kalimat1_dengan_kalimat3.append(kata)

    # user_answer = ' '.join(kalimat1_dengan_kalimat3)
    # print("\nKalimat 1 dengan sinonim yang berasal dari kalimat 3:")
    # print(user_answer)

    # kalimat1_dengan_kalimat4 = []
    # for kata in kata_kata1:
    #     sinonim_kata1 = get_synonyms(kata)
    #     if kata in kata_kata4:
    #         kalimat1_dengan_kalimat4.append(kata)
    #     elif sinonim_kata1.intersection(set(kata_kata4)):
    #         kalimat1_dengan_kalimat4.append(list(sinonim_kata1.intersection(set(kata_kata4)))[0])
    #     else:
    #         kalimat1_dengan_kalimat4.append(kata)

    # user_answer = ' '.join(kalimat1_dengan_kalimat4)
    # print("\nKalimat 1 dengan sinonim yang berasal dari kalimat 4:")
    # print(user_answer)
    
    # kalimat1_dengan_kalimat5 = []
    # for kata in kata_kata1:
    #     sinonim_kata1 = get_synonyms(kata)
    #     if kata in kata_kata5:
    #         kalimat1_dengan_kalimat5.append(kata)
    #     elif sinonim_kata1.intersection(set(kata_kata5)):
    #         kalimat1_dengan_kalimat5.append(list(sinonim_kata1.intersection(set(kata_kata5)))[0])
    #     else:
    #         kalimat1_dengan_kalimat5.append(kata)

    # user_answer = ' '.join(kalimat1_dengan_kalimat5)
    # print("\nKalimat 1 dengan sinonim yang berasal dari kalimat 5:")
    # print(user_answer)

    # Kalimat yang diproses
    sentence1 = user_answer
    sentence2 = esay_answer
    sentence3 = esay_answer2
    sentence4 = esay_answer3
    sentence5 = esay_answer4

    # Mengubah kata menjadi huruf kecil
    sentence1 = sentence1.lower()
    sentence2 = sentence2.lower()
    sentence3 = sentence3.lower()
    sentence4 = sentence4.lower()
    sentence5 = sentence5.lower()

    # Fungsi untuk menghapus tanda baca yang tidak penting
    def remove_punctuation(text):
        return re.sub(r'[^\w\s+=<>*&%-]', '', text)

    # Menghapus tanda baca yang tidak penting dari setiap kalimat
    sentence1 = remove_punctuation(sentence1)
    sentence2 = remove_punctuation(sentence2)
    sentence3 = remove_punctuation(sentence3)
    sentence4 = remove_punctuation(sentence4)
    sentence5 = remove_punctuation(sentence5)


    # Menghapus kata tidak penting
    # factory = StopWordRemoverFactory()
    # stopwords = factory.create_stop_word_remover()
    # sentence1 = stopwords.remove(sentence1)
    # sentence2 = stopwords.remove(sentence2)
    # sentence3 = stopwords.remove(sentence3)
    # sentence4 = stopwords.remove(sentence4)
    # sentence5 = stopwords.remove(sentence5)

    # Mengubah kata menjadi bentuk dasar
    # Fact = StemmerFactory()
    # Stemmer = Fact.create_stemmer()

    # sentence1 = Stemmer.stem(sentence1)
    # sentence2 = Stemmer.stem(sentence2)
    # sentence3 = Stemmer.stem(sentence3)
    # sentence4 = Stemmer.stem(sentence4)
    # sentence5 = Stemmer.stem(sentence5)

    # Inisialisasi stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Memisahkan tanda baca dan kata-kata menggunakan tokenizer dari nltk
    tokens1 = nltk.word_tokenize(sentence1)
    tokens2 = nltk.word_tokenize(sentence2)
    tokens3 = nltk.word_tokenize(sentence3)
    tokens4 = nltk.word_tokenize(sentence4)
    tokens5 = nltk.word_tokenize(sentence5)

    # Melakukan stemming hanya pada kata-kata
    stemmed_tokens1 = [stemmer.stem(token) if token.isalpha() else token for token in tokens1]
    stemmed_tokens2 = [stemmer.stem(token) if token.isalpha() else token for token in tokens2]
    stemmed_tokens3 = [stemmer.stem(token) if token.isalpha() else token for token in tokens3]
    stemmed_tokens4 = [stemmer.stem(token) if token.isalpha() else token for token in tokens4]
    stemmed_tokens5 = [stemmer.stem(token) if token.isalpha() else token for token in tokens5]

    # Menggabungkan kembali kata-kata yang telah distem ke dalam teks, sambil mempertahankan tanda baca
    sentence1 = ' '.join(stemmed_tokens1)
    sentence2 = ' '.join(stemmed_tokens2)
    sentence3 = ' '.join(stemmed_tokens3)
    sentence4 = ' '.join(stemmed_tokens4)
    sentence5 = ' '.join(stemmed_tokens5)

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    
    # Tokenize sentences
    encoded_input1 = tokenizer(sentence1, padding=True, truncation=True, return_tensors='pt')
    encoded_input2 = tokenizer(sentence2, padding=True, truncation=True, return_tensors='pt')
    encoded_input3 = tokenizer(sentence3, padding=True, truncation=True, return_tensors='pt')
    encoded_input4 = tokenizer(sentence4, padding=True, truncation=True, return_tensors='pt')
    encoded_input5 = tokenizer(sentence5, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings for each sentence
    with torch.no_grad():
        model_output1 = model(**encoded_input1)
        model_output2 = model(**encoded_input2)
        model_output3 = model(**encoded_input3)
        model_output4 = model(**encoded_input4)
        model_output5 = model(**encoded_input5)

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # Perform pooling. In this case, average pooling
    sentence_embeddings1 = mean_pooling(model_output1, encoded_input1['attention_mask'])
    sentence_embeddings2 = mean_pooling(model_output2, encoded_input2['attention_mask'])
    sentence_embeddings3 = mean_pooling(model_output3, encoded_input3['attention_mask'])
    sentence_embeddings4 = mean_pooling(model_output4, encoded_input4['attention_mask'])
    sentence_embeddings5 = mean_pooling(model_output5, encoded_input5['attention_mask'])

    # print("Sentence Embedding 1:", sentence_embeddings1)
    # print("Sentence Embedding 2:", sentence_embeddings2)
    # print("Sentence Embedding 3:", sentence_embeddings3)
    # print("Sentence Embedding 4:", sentence_embeddings4)
    # print("Sentence Embedding 5:", sentence_embeddings5)


    # Compute cosine-similarities
    # cosine_scores = util.cos_sim(sentence_embeddings1, sentence_embeddings2)
    # print("Cosine Similarity:", cosine_scores[0][0].item())

    # cosine_scores = util.cos_sim(sentence_embeddings1, sentence_embeddings3)
    # print("Cosine Similarity:", cosine_scores[0][0].item())

    # cosine_scores = util.cos_sim(sentence_embeddings1, sentence_embeddings4)
    # print("Cosine Similarity:", cosine_scores[0][0].item())

    # cosine_scores = util.cos_sim(sentence_embeddings1, sentence_embeddings5)
    # print("Cosine Similarity:", cosine_scores[0][0].item())

    # Daftar vektor embedding
    sentence_embeddings_list = [sentence_embeddings2, sentence_embeddings3, sentence_embeddings4, sentence_embeddings5]

    # Variabel untuk menyimpan nilai cosine similarity tertinggi dan indeksnya
    max_cosine_similarity = float('-inf')
    max_cosine_similarity_index = None

    # Dictionary untuk menyimpan hasil cosine similarity
    cosine_similarities = {}

    # Compute and store cosine similarities for each embedding
    for i, embeddings in enumerate(sentence_embeddings_list, start=2):
        cosine_scores = util.cos_sim(sentence_embeddings1, embeddings)
        cosine_similarity = cosine_scores[0][0].item()
        cosine_similarities[f'sentence_embeddings{i}'] = cosine_similarity
        
        # Memeriksa apakah nilai cosine similarity saat ini lebih tinggi dari nilai maksimum sebelumnya
        if cosine_similarity > max_cosine_similarity:
            max_cosine_similarity = cosine_similarity
            max_cosine_similarity_index = i
            

    print(f"Cosine similarity {cosine_similarities}")   
    print(f"Max Cosine similarity {max_cosine_similarity_index}: { max_cosine_similarity}")


    def normalize_cosine_similarity(cosine_similarity):
        # Rentang awal dari cosine similarity (biasanya 0 sampai 1)
        X_min = 0
        X_max = 1
        
        # Rentang tujuan yang diinginkan
        new_min = 0
        new_max = 20
        
        # Menghitung nilai yang dinormalisasi
        normalized_value = (max_cosine_similarity - X_min) / (X_max - X_min) * (new_max - new_min) + new_min
        
        # Bulatkan nilai yang sudah dinormalisasikan
        rounded_normalized_value = round(normalized_value)
        
        return rounded_normalized_value

    normalized_value = normalize_cosine_similarity(max_cosine_similarity)

    print(f"Nilai yang sudah dinormalisasikan dan dibulatkan ke rentang [0, 20]: {normalized_value}")


    # def mae(y_true, predictions):
    #     y_true, predictions = np.array(y_true), np.array(predictions)
    #     return np.mean(np.abs(y_true - predictions))

    # def mse(y_true, predictions):
    #     y_true, predictions = np.array(y_true), np.array(predictions)
    #     return np.mean((y_true - predictions) ** 2) 

    # def pearson_correlation(y_true, predictions):
        # Hitung koefisien korelasi Pearson menggunakan np.corrcoef
        # correlation_matrix = np.corrcoef(y_true, predictions)
        
        # Ambil elemen di baris pertama dan kolom kedua (karena kita ingin mendapatkan korelasi antara y_true dan predictions)
        # correlation_coefficient = correlation_matrix[0, 1]
        
        # return correlation_coefficient

    
    def mae(y_true, predictions):
        # Menghitung Mean Absolute Error
        absolute_errors = np.abs(np.subtract(y_true, predictions))
        return np.mean(absolute_errors)

    def mse(y_true, predictions):
        # Menghitung Mean Squared Error
        squared_errors = np.square(np.subtract(y_true, predictions))
        return np.mean(squared_errors)

    def pearson_correlation(y_true, predictions):
        # Calculate the mean of y_true and predictions
        mean_y_true = np.mean(y_true)
        mean_predictions = np.mean(predictions)
        
        # Calculate the covariance
        covariance = np.mean((y_true - mean_y_true) * (predictions - mean_predictions))
        
        # Calculate the standard deviation of y_true and predictions
        std_y_true = np.std(y_true)
        std_predictions = np.std(predictions)
        
        # Calculate the Pearson correlation coefficient
        correlation_coefficient = covariance / (std_y_true * std_predictions)
        
        return correlation_coefficient
    
    # true = [sentence_embeddings2, sentence_embeddings3, sentence_embeddings4, sentence_embeddings5]
    # predicted = [sentence_embeddings1, sentence_embeddings1, sentence_embeddings1, sentence_embeddings1]

    true = np.array([sentence_embeddings2, sentence_embeddings3, sentence_embeddings4, sentence_embeddings5])
    predicted = np.array([sentence_embeddings1, sentence_embeddings1, sentence_embeddings1, sentence_embeddings1])

    for i in range(len(true)):
        mae_value = mae(true[i], predicted[i])
        mse_value = mse(true[i], predicted[i])
        correlation_coefficient = pearson_correlation(true[i], predicted[i])
        print(f'MAE for pair {i+2}: {mae_value}')
        print(f'MSE for pair {i+2}: {mse_value}')
        print(f'Koefisien Korelasi Pearson {i+2}: {correlation_coefficient}')

    return JsonResponse({
        # 'output': cosine_scores[0][0].item()
        'output': normalized_value
    }, status=200)

@csrf_exempt
def dynamic_compile(request):
    """
    Endpoint untuk meng-compile dan menjalankan kode Java dengan test cases dinamis
    """
    user_dir = request.POST["user"]
    code = request.POST["code"]
    
    # Get question data if available
    question_data = {}
    question_id = request.POST.get("question_id")
    
    if question_id:
        try:
            import json
            import requests
            
            # Fetch question data from Laravel backend
            api_url = f"http://localhost:3001/api/questions/{question_id}/test-data"
            response = requests.get(api_url)
            
            if response.status_code == 200:
                question_data = response.json()
            else:
                print(f"Failed to fetch question data: {response.status_code}")
        except Exception as e:
            print(f"Error fetching question data: {str(e)}")
      # Run with dynamic test cases and validation
    java_runner = JavaRunner(user_directory=user_dir, code=code, question_data=question_data)
    
    # Use strict validation if question_data contains validation requirements
    has_validation_requirements = (question_data and 
                                 (question_data.get('variable_requirements') or 
                                  question_data.get('calculation_requirements') or 
                                  question_data.get('required_outputs')))
    
    if has_validation_requirements:
        print(f"Using strict validation in dynamic_compile: requirements found")
        output = java_runner.run_with_validation()
    else:
        print(f"Using standard run in dynamic_compile: no validation requirements")
        output = java_runner.run()
    
    # Add test summary if available
    if question_data and "test_cases" in question_data:
        num_tests = len(question_data["test_cases"])
        passed = 1 if output["point"] > 0 else 0
        test_summary = f"PASSED {passed}/{num_tests} TESTS"
        output["test_summary"] = test_summary
        
        # Calculate score based on percentage of tests passed
        if passed > 0:
            percentage = int(100 * passed / num_tests)
            output["point"] = max(10, percentage)  # Minimum 10 points if any test passed
    
    return JsonResponse({
        'output': output
    }, status=200)

@csrf_exempt
def dynamic_compile_llm(request):
    """
    Endpoint untuk dynamic compilation dengan test cases dari LLM
    """
    user_dir = request.POST["user"]
    code = request.POST["code"]
    
    # Get question data if available
    question_data = None
    question_id = request.POST.get("question_id")
    
    if question_id:
        try:
            import requests
            import json
            
            # Get question test data from Laravel API
            api_url = "http://localhost:3001/api/questions/{}/test-data".format(question_id)
            response = requests.get(api_url)
            
            if response.status_code == 200:
                question_data = response.json()
                # Log that we successfully got test data
                print(f"Successfully got test data for question {question_id}")
        except Exception as e:
            print(f"Error getting question data: {str(e)}")
    
    # Run with dynamic test cases
    java_runner = JavaRunner(user_directory=user_dir, code=code, question_data=question_data)
    res = java_runner.run()
    
    # Add additional information for frontend
    if res.get("test_output"):
        # Parse test output to count passed tests
        test_output = res["test_output"]
        tests_passed = test_output.count("OK")
        tests_total = tests_passed + test_output.count("FAILED")
        
        res["tests_passed"] = tests_passed
        res["tests_total"] = tests_total
    
    return JsonResponse({
        'output': res
    }, status=200)

@csrf_exempt
def dynamic_test(request):
    """
    Enhanced dynamic test endpoint dengan better error handling
    Updated untuk sistem LLM baru dengan question_id-based JUnit loading
    """
    import json
    
    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
            user_dir = data.get("user", "test_user")
            code = data.get("code", "")
            question_id = data.get("question_id")
        else:
            user_dir = request.POST.get("user", "test_user")
            code = request.POST.get("code", "")
            question_id = request.POST.get("question_id")
            
        print(f"Processing code for user: {user_dir}")
        print(f"Question ID: {question_id}")
        print(f"Code length: {len(code)} characters")
            
    except Exception as e:
        print(f"❌ Error parsing request: {e}")
        return JsonResponse({
            'output': {
                'java': f'Error parsing request: {str(e)}',
                'test_output': 'Invalid request format',
                'point': 0
            }
        }, status=400)
    
    if not code or code.strip() == "":
        return JsonResponse({
            'output': {
                'java': 'No code provided',
                'test_output': 'No Java code to execute',
                'point': 0
            }
        }, status=400)
    
    try:
        # Create JavaRunner instance with question_data for question_id
        question_data = {'question_id': question_id} if question_id else None
        java_runner = JavaRunner(user_directory=user_dir, code=code, question_data=question_data)
        
        # Set question_id for file-based JUnit loading
        if question_id:
            java_runner.question_id = int(question_id)
            print(f"🔗 Set question_id: {question_id} for JUnit file loading")
        
        # Run the code
        print("🚀 Running Java code...")
        res = java_runner.run()
        
        print(f"✅ Execution completed with point: {res.get('point', 0)}")
        return JsonResponse({'output': res}, status=200)
        
    except Exception as e:
        print(f"❌ Error running Java code: {e}")
        return JsonResponse({
            'output': {
                'java': f'Error executing code: {str(e)}',
                'test_output': 'Execution failed',
                'point': 0
            }
        }, status=500)
    
@csrf_exempt
def generate_question_with_junit(request):
    """
    Generate question with JUnit test - HANYA UNTUK PREVIEW, TIDAK MENYIMPAN FILE
    """
    try:
        import json
        from llm_services.services.llm_service import LLMService
        
        # Handle both GET and POST requests
        if request.method == 'GET':
            content_id = request.GET.get('content_id')
        elif request.method == 'POST':
            data = json.loads(request.body)
            content_id = data.get('content_id')
        else:
            return JsonResponse({"error": "Method not allowed"}, status=405)
        
        if not content_id:
            return JsonResponse({"error": "content_id is required"}, status=400)
        
        print(f"🔥 Generating question preview for content_id: {content_id}")
        
        # Initialize LLM service
        llm_service = LLMService()
        
        # Generate question with tests and history tracking
        print("🤖 Calling LLM service for question generation with tests...")
        result = llm_service.generate_question_with_tests_django(int(content_id))
        
        if "error" in result:
            print(f"❌ LLM generation error: {result['error']}")
            return JsonResponse({
                "success": False,
                "error": f"LLM generation failed: {result['error']}"
            }, status=500)
        
        print(f"✅ Question generation completed in {result.get('generation_time', 0):.2f}s")
        print(f"🎯 History ID: {result.get('history_id', 'N/A')} (Full tracking enabled)")
        
        # Prepare response for frontend
        return JsonResponse({
            "success": True,
            "data": {
                "studi_kasus": result.get("studi_kasus", ""),
                "tugas": result.get("tugas", ""),
                "code": result.get("code", ""),
                "test_cases": result.get("test_cases", []),
                "junit_test_code": result.get("junit_test_code", ""),
                "solution_template": result.get("solution_template", ""),
                "class_name": result.get("class_name", "GeneratedClass"),
                "generation_time": result.get("generation_time", 0),
                "content_id": content_id,
                "history_id": result.get("history_id", None),  # Track history for debugging
                "tracking_enabled": True  # Indicate full tracking is enabled
            }
        })
        
    except Exception as e:
        print(f"❌ Error in generate_question_with_junit: {str(e)}")
        import traceback
        traceback.print_exc()
        return JsonResponse({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }, status=500)

@csrf_exempt 
def save_junit_test_file(request):
    """
    Save JUnit test file setelah question berhasil disimpan
    Dipanggil dari Laravel setelah form submission berhasil
    """
    if request.method != 'POST':
        return JsonResponse({"error": "Method not allowed"}, status=405)
        
    try:
        import json
        from llm_services.services.llm_service import LLMService
        
        print(f"🔥 save_junit_test_file endpoint called")
        print(f"   - Request method: {request.method}")
        print(f"   - Content type: {request.content_type}")
        
        data = json.loads(request.body)
        junit_test_code = data.get('junit_test_code')
        class_name = data.get('class_name')
        question_id = data.get('question_id')
        
        print(f"   - Received data:")
        print(f"     * question_id: {question_id}")
        print(f"     * class_name: {class_name}")
        print(f"     * junit_test_code length: {len(junit_test_code) if junit_test_code else 0}")
        
        if not all([junit_test_code, class_name, question_id]):
            print(f"❌ Missing required fields")
            return JsonResponse({
                "error": "junit_test_code, class_name, and question_id are required"
            }, status=400)
        
        print(f"🔄 Saving JUnit file for question_id: {question_id}")
        
        # Initialize LLM service
        llm_service = LLMService()
        
        # Save JUnit test file
        junit_file_path = llm_service.save_junit_test_file(
            junit_test_code, class_name, question_id
        )
        
        if junit_file_path:
            print(f"✅ JUnit file saved: {junit_file_path}")
            return JsonResponse({
                "success": True,
                "junit_file_path": junit_file_path,
                "message": f"JUnit test file successfully saved for question {question_id}"
            })
        else:
            print(f"❌ Failed to save JUnit file")
            return JsonResponse({
                "success": False,
                "error": "Failed to save JUnit test file"
            }, status=500)
        
    except Exception as e:
        print(f"❌ Error in save_junit_test_file: {str(e)}")
        import traceback
        traceback.print_exc()
        return JsonResponse({
            "error": f"Failed to save JUnit test file: {str(e)}"
        }, status=500)


def extract_class_name_from_code(code: str) -> str:
    """Extract class name from Java code"""
    if not code:
        return None
        
    # Look for class declaration
    import re
    class_pattern = r'public\s+class\s+(\w+)'
    match = re.search(class_pattern, code)
    if match:
        return match.group(1)
        
    # Fallback: look for any class declaration
    class_pattern = r'class\s+(\w+)'
    match = re.search(class_pattern, code)
    if match:
        return match.group(1)
        
    return None


def save_junit_test_to_file(junit_code: str, class_name: str, question_id: int) -> str:
    """
    Save JUnit test code to test_cases directory with proper naming convention
    
    Args:
        junit_code: JUnit test code string
        class_name: Java class name
        question_id: Question ID for filename
    
    Returns:
        Path to saved file or None if failed
    """
    try:
        # Ensure test_cases directory exists
        test_cases_dir = "java_files/test_cases"
        os.makedirs(test_cases_dir, exist_ok=True)
        
        # Create filename with pattern expected by JavaRunner
        junit_filename = f"JUnit{class_name}Test_Q{question_id}.java"
        file_path = os.path.join(test_cases_dir, junit_filename)
        
        # Process JUnit code: ensure it has proper package placeholder
        processed_junit_code = process_junit_code_for_file(junit_code, class_name)
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(processed_junit_code)
        
        print(f"✅ JUnit test saved: {file_path}")
        return file_path
        
    except Exception as e:
        print(f"❌ Error saving JUnit test: {e}")
        return None


def process_junit_code_for_file(junit_code: str, class_name: str) -> str:
    """
    Process JUnit code to ensure proper format for file system
    
    Args:
        junit_code: Raw JUnit code from LLM
        class_name: Java class name
    
    Returns:
        Processed JUnit code ready for file
    """
    import re
    
    # Remove any existing package declaration
    junit_code = re.sub(r'package\s+[^;]+;', '', junit_code)
    
    # Don't add package declaration for file system usage - it will be handled when creating files
    processed_code = junit_code.strip()
    
    # Ensure proper class name in JUnit test - match with filename
    # Replace any existing test class names with the expected format
    processed_code = re.sub(
        r'public\s+class\s+\w*Test\s*\{',
        f'public class {class_name}Test {{',
        processed_code
    )
    
    # Replace any placeholder class references
    processed_code = processed_code.replace("{{class_name}}", class_name)
    
    return processed_code

# History endpoints untuk menggantikan FastAPI history_router.py
@csrf_exempt
def get_history_list(request):
    """
    Get history list with pagination - Django version (migrated from FastAPI)
    """
    if request.method != 'GET':
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    try:
        limit = int(request.GET.get('limit', 20))
        offset = int(request.GET.get('offset', 0))
        
        # Validate parameters
        limit = max(1, min(limit, 100))
        offset = max(0, offset)
        
        print(f"📜 Fetching history list - limit: {limit}, offset: {offset}")
        
        # Try to use the migrated GenerationHistoryModel
        from llm_services.models.generation_history import GenerationHistoryModel
        history_model = GenerationHistoryModel()
        history_list = history_model.get_history_list(limit=limit, offset=offset)
        total_count = history_model.get_total_count()
            
        return JsonResponse({
            "success": True,
            "data": history_list,
            "total": total_count,
            "limit": limit,
            "offset": offset
        })
            
        
    except Exception as e:
        print(f"❌ Error fetching history list: {str(e)}")
        return JsonResponse({
            "success": False,
            "error": f"Error fetching history list: {str(e)}"
        }, status=500)

@csrf_exempt
def get_history_detail(request, history_id):
    """
    Get history detail by ID - Django version (migrated from FastAPI)
    """
    if request.method != 'GET':
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    try:
        print(f"📖 Fetching history detail for ID: {history_id}")
        
        from llm_services.models.generation_history import GenerationHistoryModel
        history_model = GenerationHistoryModel()
        history_detail = history_model.get_history_by_id(history_id)
        
        if history_detail:
            return JsonResponse({
                "success": True,
                "data": history_detail
            })
        else:
            return JsonResponse({
                "success": False,
                "error": f"History with ID {history_id} not found"
            }, status=404)
        
    except Exception as e:
        print(f"❌ Error fetching history detail: {str(e)}")
        return JsonResponse({
            "success": False,
            "error": f"Error fetching history detail: {str(e)}"
        }, status=500)

@csrf_exempt
def get_content_history(request, content_id):
    """
    Get history for specific content - Django version (migrated from FastAPI)
    """
    if request.method != 'GET':
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    try:
        limit = int(request.GET.get('limit', 10))
        limit = max(1, min(limit, 50))
        
        print(f"📋 Fetching content history for content_id: {content_id}, limit: {limit}")
        
        # Try to use the migrated GenerationHistoryModel
        from llm_services.models.generation_history import GenerationHistoryModel
        history_model = GenerationHistoryModel()
        content_histories = history_model.get_history_by_content_id(content_id, limit)
        
        return JsonResponse({
            "success": True,
            "data": content_histories
        })
        
    except Exception as e:
        print(f"❌ Error fetching content history: {str(e)}")
        return JsonResponse({
            "success": False,
            "error": f"Error fetching content history: {str(e)}"
        }, status=500)

@csrf_exempt
def search_history(request):
    """
    Search history by topic title - Django version (migrated from FastAPI)
    """
    if request.method != 'GET':
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    try:
        search_term = request.GET.get('q', '').strip()
        limit = int(request.GET.get('limit', 20))
        offset = int(request.GET.get('offset', 0))
        
        # Validate parameters
        limit = max(1, min(limit, 100))
        offset = max(0, offset)
        
        print(f"🔍 Searching history - term: '{search_term}', limit: {limit}, offset: {offset}")
        
        if not search_term:
            return JsonResponse({
                "success": True,
                "data": []
            })
        

        from llm_services.models.generation_history import GenerationHistoryModel
        history_model = GenerationHistoryModel()
        search_results = history_model.search_history(search_term, limit, offset)
        total_results = history_model.get_search_total_count(search_term)
        
        return JsonResponse({
            "success": True,
            "data": search_results,
            "search_term": search_term,
            "total": total_results,
            "limit": limit,
            "offset": offset
        })
            
        
    except Exception as e:
        print(f"❌ Error searching history: {str(e)}")
        return JsonResponse({
            "success": False,
            "error": f"Error searching history: {str(e)}"
        }, status=500)

@csrf_exempt
def dynamic_test_with_database_junit(request):
    """
    Run student code against JUnit test stored in database
    Updated to maintain user package system consistency
    """
    try:
        import json
        import os
        import tempfile
        import shutil
        from django.http import JsonResponse
        import requests
        from lib.java_runner import JavaRunner
        
        if request.method != 'POST':
            return JsonResponse({"error": "Method not allowed"}, status=405)
        
        data = json.loads(request.body)
        question_id = data.get('question_id')
        student_code = data.get('code', '')
        user_email = data.get('user', 'test_user@example.com')  # Get user from request
        
        if not question_id:
            return JsonResponse({"error": "question_id is required"}, status=400)
        
        if not student_code.strip():
            return JsonResponse({"error": "Student code is required"}, status=400)
        
        print(f"🔥 Running dynamic test with database JUnit for question_id: {question_id}")
        print(f"👤 User: {user_email}")
        
        # Fetch JUnit test from Laravel database
        junit_data = fetch_junit_from_database(question_id)
        if not junit_data:
            return JsonResponse({
                "error": f"No JUnit test found for question_id: {question_id}"
            }, status=404)
        
        junit_code = junit_data.get('junit_code')
        class_name = junit_data.get('class_name', 'GeneratedClass')
        test_cases = junit_data.get('test_cases', [])
        
        # [UPDATED] Allow empty JUnit if test_cases exist (I/O Grading Support)
        if not junit_code and not test_cases:
            return JsonResponse({
                "error": f"Both JUnit code and Test Cases are empty for question_id: {question_id}"
            }, status=400)
        
        if not junit_code and test_cases:
            print(f"ℹ️ using I/O based grading for question_id: {question_id}")
        
        # Use JavaRunner to maintain user package system consistency
        # Prepare question_data with JUnit information
        question_data = {
            'question_id': question_id,
            'junit_tests': junit_code,
            'test_cases': test_cases, # Pass test cases to runner
            'class_name': class_name,
            'use_database_junit': bool(junit_code),
            'score': junit_data.get('score', 30)
        }
        
        try:
            # Create JavaRunner instance with user directory and question data
            java_runner = JavaRunner(user_directory=user_email, code=student_code, question_data=question_data)
            java_runner.question_id = int(question_id)
            
            print(f"🏗️ Using JavaRunner with user directory for consistency")
            
            # Run with validation (includes JUnit testing)
            result = java_runner.run_with_validation()
            
            # [AI Grading Integration]
            # If score < 100 but executed successfully, try AI Grading
            if result.get('point', 0) < 100 \
               and "COMPILATION FAILED!" not in result.get('test_output', '') \
               and "RUNTIME ERROR!" not in result.get('test_output', '') \
               and result.get('details'): # Check if detailed test results exist
               
                print(f"🤖 Attempts AI Grading Fallback for QID {question_id}...")
                try:
                     from llm_services.services.llm_service import LLMService
                     llm_service = LLMService()
                     ai_result = llm_service.grade_submission(
                        student_code=student_code, 
                        test_results=result.get('details'),
                        question_text=f"Question ID {question_id}"
                     )
                     
                     if ai_result.get('is_correct') and ai_result.get('score', 0) > result.get('point', 0):
                        print(f"✅ AI Judge Overruled! New Score: {ai_result.get('score')}")
                        result['point'] = ai_result.get('score')
                        result['can_submit'] = ai_result.get('score') == 100
                        result['java'] += f"\n\n🤖 AI Judge Feedback: {ai_result.get('feedback')}"
                     else:
                        print(f"🤖 AI Judge Agreed with Runner. Score remains: {result.get('point')}")
                except Exception as e:
                     print(f"⚠️ AI Grading Failed: {e}")

            print(f"✅ Test execution completed for question_id: {question_id}")
            print(f"🔍 Result details: {result}")
            
            # Check if the execution was successful (not a system error)
            is_execution_successful = (
                "COMPILATION FAILED!" not in result.get('test_output', '') and
                "RUNTIME ERROR!" not in result.get('test_output', '') and
                "SYSTEM ERROR!" not in result.get('test_output', '')
            )
            
            # Format response to match expected format
            if is_execution_successful:
                # Determine status based on test results and can_submit flag
                can_submit = result.get('can_submit', False)
                point = result.get('point', 0)
                
                # Prepare detailed output
                java_output = result.get('java', '')
                test_output = result.get('test_output', '')
                
                # Combine outputs with clear separation
                full_output = ""
                if java_output:
                    full_output += f"Program Output:\n{java_output}\n"
                
                if test_output:
                    full_output += f"\nTest Results:\n{test_output}"
                
                if can_submit and point > 0:
                    status = 'success'
                    passed = True
                    message = f"All tests passed! Score: {point}"
                    full_output += f"\n\n✅ SUCCESS: All JUnit tests passed successfully!"
                else:
                    status = 'test_failed' 
                    passed = False
                    message = "Tests failed - please check your code"
                    full_output += f"\n\n❌ FAILED: Some JUnit tests failed"
                
                return JsonResponse({
                    "success": True,
                    "result": {
                        "status": status,
                        "output": full_output,
                        "message": message,
                        "passed": passed,
                        "point": point,
                        "can_submit": can_submit
                    },
                    "question_id": question_id,
                    "class_name": class_name
                })
            else:
                # Compilation or runtime error occurred
                error_details = result.get('test_output', 'Execution failed')
                java_output = result.get('java', '')
                
                # Create detailed error output
                error_output = ""
                if java_output:
                    error_output += f"Program Output:\n{java_output}\n"
                
                error_output += f"\n❌ ERROR DETAILS:\n{error_details}"
                
                # Determine specific error type and provide helpful message
                if "COMPILATION FAILED!" in error_details:
                    error_message = "Code compilation failed - please check syntax errors"
                    error_output += f"\n\n💡 HELP: Check for syntax errors like missing semicolons, brackets, or typos in your code"
                elif "RUNTIME ERROR!" in error_details:
                    error_message = "Runtime error occurred - please check your logic"
                    error_output += f"\n\n💡 HELP: Check for runtime issues like null pointer exceptions, array bounds, or logic errors"
                elif "SYSTEM ERROR!" in error_details:
                    error_message = "System error occurred during execution"
                    error_output += f"\n\n💡 HELP: This is a system issue, please try again or contact support"
                else:
                    error_message = "Execution failed - please check your code"
                    error_output += f"\n\n💡 HELP: Review the error details above and fix any issues in your code"
                
                return JsonResponse({
                    "success": True,  # Still return success=True so frontend shows the detailed error
                    "result": {
                        "status": "compile_error",
                        "error": error_details,
                        "message": error_message,
                        "output": error_output,
                        "passed": False,
                        "point": 0,
                        "can_submit": False
                    },
                    "question_id": question_id,
                    "class_name": class_name
                })
                
        except Exception as runner_error:
            print(f"❌ JavaRunner execution failed: {str(runner_error)}")
            
            # Fallback to temporary directory method if JavaRunner fails
            print("🔄 Falling back to temporary directory method...")
            
            # Convert user email to package format
            user_package = user_email.replace('@', '_').replace('.', '_')
            print(f"📦 Using user package: {user_package}")
            
            # Create temporary directory for test execution
            temp_dir = None
            try:
                temp_dir = tempfile.mkdtemp(prefix='junit_test_')
                print(f"📁 Created temp directory: {temp_dir}")
                
                # Create JUnit test file with user package
                junit_file_path = create_junit_test_file(temp_dir, junit_code, class_name, user_package)
                
                # Create student code file with user package
                student_file_path = create_student_code_file(temp_dir, student_code, class_name, user_package)
                
                # Run JUnit test
                result = run_junit_test(temp_dir, class_name)
                
                print(f"✅ Fallback test execution completed for question_id: {question_id}")
                
                return JsonResponse({
                    "success": True,
                    "result": result,
                    "question_id": question_id,
                    "class_name": class_name
                })
                
            finally:
                # Clean up temporary directory
                if temp_dir and os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                        print(f"🗑️ Cleaned up temp directory: {temp_dir}")
                    except Exception as cleanup_error:
                        print(f"⚠️ Failed to cleanup temp directory: {cleanup_error}")
        
    except Exception as e:
        print(f"❌ Error in dynamic_test_with_database_junit: {str(e)}")
        import traceback
        traceback.print_exc()
        return JsonResponse({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }, status=500)

def fetch_junit_from_database(question_id):
    """
    Fetch JUnit test code from Laravel database
    """
    try:
        import requests
        from django.conf import settings
        
        # Laravel API URL to fetch question data
        base_url = getattr(settings, 'LARAVEL_API_BASE_URL', 'http://127.0.0.1:8001')
        laravel_api_url = f"{base_url}/api/questions/{question_id}/junit"
        
        print(f"🌐 Fetching JUnit from Laravel API: {laravel_api_url}")
        
        response = requests.get(laravel_api_url)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                return {
                    'junit_code': data.get('junit_tests', ''),
                    'class_name': data.get('class_name', 'GeneratedClass'),
                    'solution_template': data.get('solution_template', ''),
                    'test_cases_data': data.get('test_cases_data', ''),
                    'test_cases': data.get('test_cases', []), # Get test cases list
                }
        
        print(f"❌ Failed to fetch JUnit from Laravel API: {response.status_code}")
        return None
        
    except Exception as e:
        print(f"❌ Error fetching JUnit from database: {str(e)}")
        return None

def create_junit_test_file(temp_dir, junit_code, class_name, user_package=None):
    """
    Create JUnit test file in temporary directory
    """
    try:
        import re
        
        test_file_path = os.path.join(temp_dir, f"{class_name}Test.java")
        
        # Replace the {{user_package}} placeholder with actual user package
        if user_package:
            # Use the user package (email converted)
            processed_junit_code = junit_code.replace("{{user_package}}", user_package)
            processed_junit_code = processed_junit_code.replace("{{USER_PACKAGE}}", user_package) # Handle uppercase variant
            # Ensure package declaration exists
            if not processed_junit_code.strip().startswith("package"):
                processed_junit_code = f"package {user_package};\n\n" + processed_junit_code
        else:
            # Remove package declaration if no user package provided
            processed_junit_code = junit_code.replace("{{user_package}}", "")
            processed_junit_code = processed_junit_code.replace("{{USER_PACKAGE}}", "") # Handle uppercase variant
            # Remove empty package declaration lines
            processed_junit_code = re.sub(r'package\s*;\s*\n?', '', processed_junit_code)
        
        # Ensure the class name in the code matches the filename
        # This is critical for Java compilation
        processed_junit_code = re.sub(
            r'public\s+class\s+\w*Test\s*\{',
            f'public class {class_name}Test {{',
            processed_junit_code
        )
        
        # Clean up any extra newlines at the beginning
        processed_junit_code = processed_junit_code.lstrip()
        
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(processed_junit_code)
        
        print(f"📝 Created JUnit test file: {test_file_path}")
        return test_file_path
        
    except Exception as e:
        print(f"❌ Error creating JUnit test file: {str(e)}")
        raise

def create_student_code_file(temp_dir, student_code, class_name, user_package):
    """
    Create student code file in temporary directory with user package
    """
    try:
        import re
        
        student_file_path = os.path.join(temp_dir, f"{class_name}.java")
        
        # Remove any package declarations from student code to avoid conflicts
        processed_student_code = re.sub(r'package\s+[^;]+;\s*\n?', '', student_code)
        
        # Clean up any extra newlines at the beginning
        processed_student_code = processed_student_code.lstrip()
        
        # Add user package declaration at the beginning
        final_student_code = f"package {user_package};\n\n{processed_student_code}"
        
        with open(student_file_path, 'w', encoding='utf-8') as f:
            f.write(final_student_code)
        
        print(f"📝 Created student code file: {student_file_path}")
        return student_file_path
        
    except Exception as e:
        print(f"❌ Error creating student code file: {str(e)}")
        raise

def run_junit_test(temp_dir, class_name):
    """
    Run JUnit test and return results
    """
    try:
        import subprocess
        import os
        
        # Define JUnit JAR paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        junit_jar = os.path.join(project_root, 'java_files', 'junit-4.13.2.jar')
        hamcrest_jar = os.path.join(project_root, 'java_files', 'hamcrest-core-1.3.jar')
        
        # Build classpath
        classpath = f"{temp_dir}{os.pathsep}{junit_jar}{os.pathsep}{hamcrest_jar}"
        
        print(f"🔧 Compiling Java files in: {temp_dir}")
        
        # Compile both student code and test
        compile_cmd = [
            'javac', '-cp', classpath,
            os.path.join(temp_dir, f"{class_name}.java"),
            os.path.join(temp_dir, f"{class_name}Test.java")
        ]
        
        compile_result = subprocess.run(
            compile_cmd,
            cwd=temp_dir,
            capture_output=True,
            text=True
        )
        
        if compile_result.returncode != 0:
            return {
                "status": "compile_error",
                "message": "Compilation failed",
                "error": compile_result.stderr,
                "stdout": compile_result.stdout
            }
        
        print(f"✅ Compilation successful")
        
        # Run JUnit test
        test_cmd = [
            'java', '-cp', classpath,
            'org.junit.runner.JUnitCore',
            f"{class_name}Test"
        ]
        
        test_result = subprocess.run(
            test_cmd,
            cwd=temp_dir,
            capture_output=True,
            text=True
        )
        
        # Parse test results
        output = test_result.stdout + test_result.stderr
        
        if test_result.returncode == 0:
            return {
                "status": "success",
                "message": "All tests passed",
                "output": output,
                "passed": True
            }
        else:
            return {
                "status": "test_failed",
                "message": "Some tests failed",
                "output": output,
                "passed": False
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "message": "Test execution timed out",
            "output": "Test execution exceeded time limit"
        }
    except Exception as e:
        print(f"❌ Error running JUnit test: {str(e)}")
        return {
            "status": "error",
            "message": f"Test execution error: {str(e)}",
            "output": str(e)
        }

@csrf_exempt
def evaluate_rag_retrieval(request):
    """
    Endpoint for evaluating RAG retrieval performance
    Calculates Recall@k, Precision@k, and MRR metrics
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)
    
    try:
        # Initialize RAG service
        rag_service = RAGService()
        
        if not rag_service.is_available():
            return JsonResponse({
                'error': 'RAG service not available',
                'message': 'Vector store is not loaded or accessible'
            }, status=503)
        
        # Get evaluation parameters from request
        import json
        request_data = json.loads(request.body) if request.body else {}
        
        k_values = request_data.get('k_values', [1, 3, 5])
        test_queries = request_data.get('test_queries', None)
        ground_truth = request_data.get('ground_truth', None)
        
        print(f"🔍 Starting RAG evaluation with k_values: {k_values}")
        
        # Run evaluation pipeline
        if test_queries and ground_truth:
            # Use custom test set
            eval_results = rag_service.evaluate_retrieval_performance(
                test_queries=test_queries,
                ground_truth=ground_truth,
                k_values=k_values
            )
        else:
            # Use default test set
            eval_results = rag_service.evaluate_retrieval_performance(k_values=k_values)
        
        if 'error' in eval_results:
            return JsonResponse({
                'error': 'Evaluation failed',
                'message': eval_results['error']
            }, status=500)
        
        # Format response
        response_data = {
            'status': 'success',
            'message': 'RAG evaluation completed successfully',
            'evaluation_results': eval_results,
            'summary': {
                'num_queries': eval_results.get('num_queries', 0),
                'mrr_score': eval_results['metrics']['mrr']['score'],
                'metrics_by_k': {}
            }
        }
        
        # Add metrics summary for each k
        for k in k_values:
            if f'recall@{k}' in eval_results['metrics']:
                response_data['summary']['metrics_by_k'][f'k_{k}'] = {
                    'recall': eval_results['metrics'][f'recall@{k}']['mean'],
                    'precision': eval_results['metrics'][f'precision@{k}']['mean']
                }
        
        print(f"✅ RAG evaluation completed successfully")
        print(f"📊 MRR: {response_data['summary']['mrr_score']:.3f}")
        
        return JsonResponse(response_data)
        
    except Exception as e:
        print(f"❌ Error in RAG evaluation endpoint: {str(e)}")
        return JsonResponse({
            'error': 'Internal server error',
            'message': str(e)
        }, status=500)

@csrf_exempt
def run_rag_evaluation_pipeline(request):
    """
    Endpoint for running complete RAG evaluation pipeline
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)
    
    try:
        # Initialize RAG service
        rag_service = RAGService()
        
        print(f"🚀 Starting complete RAG evaluation pipeline...")
        
        # Run complete evaluation pipeline
        eval_results = rag_service.run_evaluation_pipeline()
        
        if 'error' in eval_results:
            return JsonResponse({
                'error': 'Evaluation pipeline failed',
                'message': eval_results['error']
            }, status=500)
        
        return JsonResponse({
            'status': 'success',
            'message': 'RAG evaluation pipeline completed successfully',
            'results': eval_results,
            'saved_to': eval_results.get('saved_to', None)
        })
        
    except Exception as e:
        print(f"❌ Error in RAG evaluation pipeline: {str(e)}")
        return JsonResponse({
            'error': 'Internal server error',
            'message': str(e)
        }, status=500)