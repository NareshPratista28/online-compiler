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
    esay_answer = request.POST["esay_answer"]
    user_answer = request.POST["user_answer"]


    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Kalimat yang diproses
    sentence1 = esay_answer
    sentence2 = user_answer

    # Fungsi untuk menghapus tanda baca yang tidak penting
    def remove_punctuation(text):
        return re.sub(r'[^\w\s]', '', text)

    # Menghapus tanda baca yang tidak penting dari setiap kalimat
    sentence1 = remove_punctuation(sentence1)
    sentence2 = remove_punctuation(sentence2)

    # Mengubah kata menjadi huruf kecil
    sentence1 = sentence1.lower()
    sentence2 = sentence2.lower()

    # Menghapus kata tidak penting
    factory = StopWordRemoverFactory()
    stopwords = factory.create_stop_word_remover()
    sentence1 = stopwords.remove(sentence1)
    sentence2 = stopwords.remove(sentence2)

    # Mengubah kata menjadi bentuk dasar
    Fact = StemmerFactory()
    Stemmer = Fact.create_stemmer()

    sentence1 = Stemmer.stem(sentence1)
    sentence2 = Stemmer.stem(sentence2)

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    # Tokenize sentences
    encoded_input1 = tokenizer(sentence1, padding=True, truncation=True, return_tensors='pt')
    encoded_input2 = tokenizer(sentence2, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings for each sentence
    with torch.no_grad():
        model_output1 = model(**encoded_input1)
        model_output2 = model(**encoded_input2)

    # Perform pooling. In this case, average pooling
    sentence_embeddings1 = mean_pooling(model_output1, encoded_input1['attention_mask'])
    sentence_embeddings2 = mean_pooling(model_output2, encoded_input2['attention_mask'])

    # Compute cosine-similarities
    cosine_scores = util.cos_sim(sentence_embeddings1, sentence_embeddings2)
    print("Cosine Similarity:", cosine_scores[0][0].item())

    return JsonResponse({
        'output': cosine_scores[0][0].item()
    }, status=200)