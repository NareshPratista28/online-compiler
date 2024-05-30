import os

from django.http import JsonResponse

from lib.FileUploader import FileUploader
from lib.java_runner import JavaRunner
from django.views.decorators.csrf import csrf_exempt

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
import torch
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
from nltk.corpus import wordnet

import numpy as np
from scipy.stats import pearsonr

@csrf_exempt
def index(request):
    user_dir = request.POST["user"]
    code = request.POST["code"]

    java_runner = JavaRunner(user_directory=user_dir, code=code)
    res = java_runner.run()

    return JsonResponse({
        'output': res
    }, status=200)


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

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
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

    # Fungsi untuk menghapus tanda baca yang tidak penting
    def remove_punctuation(text):
        return re.sub(r'[^\w\s+=<>*&%-]', '', text)

    # Menghapus tanda baca yang tidak penting dari setiap kalimat
    sentence1 = remove_punctuation(sentence1)
    sentence2 = remove_punctuation(sentence2)
    sentence3 = remove_punctuation(sentence3)
    sentence4 = remove_punctuation(sentence4)
    sentence5 = remove_punctuation(sentence5)


    # Mengubah kata menjadi huruf kecil
    sentence1 = sentence1.lower()
    sentence2 = sentence2.lower()
    sentence3 = sentence3.lower()
    sentence4 = sentence4.lower()
    sentence5 = sentence5.lower()


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
        'output': max_cosine_similarity
    }, status=200)