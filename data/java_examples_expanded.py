from data.java_examples_expanded_p1 import EXPANDED_JAVA_EXAMPLES_P1
from data.java_examples_expanded_p2 import EXPANDED_JAVA_EXAMPLES_P2
from data.java_examples_expanded_p3 import EXPANDED_JAVA_EXAMPLES_P3

# Menggabungkan 120 dataset tambahan (Topik 1-12)
EXPANDED_JAVA_EXAMPLES = EXPANDED_JAVA_EXAMPLES_P1 + EXPANDED_JAVA_EXAMPLES_P2 + EXPANDED_JAVA_EXAMPLES_P3

# Dictionary Helper untuk Organisasi Kategori seperti di file lama
EXPANDED_JAVA_EXAMPLES_BY_CATEGORY = {
    "Tipe Data, Variabel dan Operator": [ex for ex in EXPANDED_JAVA_EXAMPLES if ex['topic'] == "Tipe Data, Variabel dan Operator"],
    "Sintaks Pemilihan 1": [ex for ex in EXPANDED_JAVA_EXAMPLES if ex['topic'] == "Sintaks Pemilihan 1"],
    "Sintaks Pemilihan 2": [ex for ex in EXPANDED_JAVA_EXAMPLES if ex['topic'] == "Sintaks Pemilihan 2"],
    "Sintaks Pemilihan Bersarang": [ex for ex in EXPANDED_JAVA_EXAMPLES if ex['topic'] == "Sintaks Pemilihan Bersarang"],
    "Perulangan While": [ex for ex in EXPANDED_JAVA_EXAMPLES if ex['topic'] == "Perulangan While"],
    "Perulangan Do-While": [ex for ex in EXPANDED_JAVA_EXAMPLES if ex['topic'] == "Perulangan Do-While"],
    "Sintaks Perulangan 1": [ex for ex in EXPANDED_JAVA_EXAMPLES if ex['topic'] == "Sintaks Perulangan 1"],
    "Sintaks Perulangan 2": [ex for ex in EXPANDED_JAVA_EXAMPLES if ex['topic'] == "Sintaks Perulangan 2"],
    "Array 1": [ex for ex in EXPANDED_JAVA_EXAMPLES if ex['topic'] == "Array 1"],
    "Array Multidimensi": [ex for ex in EXPANDED_JAVA_EXAMPLES if ex['topic'] == "Array Multidimensi"],
    "Fungsi Static": [ex for ex in EXPANDED_JAVA_EXAMPLES if ex['topic'] == "Fungsi Static"],
    "Fungsi Rekursif": [ex for ex in EXPANDED_JAVA_EXAMPLES if ex['topic'] == "Fungsi Rekursif"]
}
