from src.preprocessing.preprocessor import Preprocessor

from mpstemmer import MPStemmer

import nltk
from nltk.corpus import stopwords

import html
import re
import time
import ssl

import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_suffix_regex, compile_infix_regex


# ========= HANYA DIJALANKAN SEKALI SAAT AWAL =========

# Perbaikan koneksi SSL untuk NLTK di lingkungan tertentu
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

# Pastikan resource NLTK tersedia
nltk_packages = ["punkt", "punkt_tab", "stopwords"]
for pkg in nltk_packages:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if pkg ==
                       "punkt" else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)

# Load Stopwords satu kali
INDONESIAN_STOPWORDS = set(stopwords.words("indonesian"))

# Inisialisasi Spacy + tokenizer khusus


def custom_tokenizer(nlp):
    infix_re = compile_infix_regex([
        r"[-~]",
    ])
    return Tokenizer(
        nlp.vocab,
        rules=nlp.Defaults.tokenizer_exceptions,
        prefix_search=compile_prefix_regex(nlp.Defaults.prefixes).search,
        suffix_search=compile_suffix_regex(nlp.Defaults.suffixes).search,
        infix_finditer=infix_re.finditer,
    )


SPACY_NLP = spacy.blank("id")
SPACY_NLP.tokenizer = custom_tokenizer(SPACY_NLP)
SPACY_NLP.add_pipe("lemmatizer", config={"mode": "lookup"})
SPACY_NLP.initialize()

# Stemmer MP lebih cepat
STEMMER_MP = MPStemmer()


# ========== KELAS PREPROCESSOR ==========

class TextPreprocessor(Preprocessor):
    nlp = SPACY_NLP
    stemmerMP = STEMMER_MP
    stop_words = INDONESIAN_STOPWORDS

    def __init__(self):
        # Inisialisasi satu kali
        pass

    def normalize_custom_words(self, text):
        replacements = {
            "rp": "rupiah", "usd": "dolar", "idr": "rupiah",
            "amp": "", "nbsp": "",
        }
        for word, replacement in replacements.items():
            text = re.sub(rf"\b{word}\b", replacement, text)
        return text

    def auto_protect_keywords(self, text):
        pattern = r'\b(?:[a-z]*\d+[a-z]+[a-z\d-]*)\b|\b(?:\d+[a-z]+)\b'
        keywords = re.findall(pattern, text.lower())
        protected_map = {}
        for i, keyword in enumerate(set(keywords)):
            token = f"PROTECTED{i}"
            protected_map[token] = keyword
            text = re.sub(rf"\b{re.escape(keyword)}\b", token, text)
        return text, protected_map

    def restore_keywords(self, text, protected_map):
        for token, keyword in protected_map.items():
            text = text.replace(token, keyword)
        return text

    def lemmatize_text(self, text):
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc])

    def preprocess(self, text):
        print(f"Text Awal: {text}")
        if not text.strip():
            return None

        text = text.lower()
        text = text.encode('utf-8', 'ignore').decode('utf-8')
        text = html.unescape(text)
        text = re.sub(r'\brp\d+([.,]\d+)*\b', 'rupiah', text)
        text = re.sub(r'\bidr\d+([.,]\d+)*\b', 'rupiah', text)
        text = self.normalize_custom_words(text)
        text = re.sub(r"&[a-z]+;", " ", text)
        text = re.sub(r"\b(?!ke-\d+)\d+\b", "", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\b(\w+)([- ]\1)+\b", r"\1", text)

        text_cleaned = text.strip()
        if len(text_cleaned) < 2:
            return text_cleaned

        tokens = [t for t in nltk.word_tokenize(
            text_cleaned) if t not in self.stop_words and len(t) > 1]
        text = " ".join(tokens) or text_cleaned

        text, protected_map = self.auto_protect_keywords(text)
        text = self.lemmatize_text(text)
        text = self.restore_keywords(text, protected_map)

        text = self.stemmerMP.stem_kalimat(text)
        tokens = [t for t in nltk.word_tokenize(text) if len(t) > 1]
        text = " ".join(tokens)

        print(f"Text Setelah: {text}")
        return text or text_cleaned


# ========== TEST RUNNER (Opsional, untuk uji coba) ==========

if __name__ == "__main__":
    preprocessor = TextPreprocessor()

    samples = [
        "Persipura Jayapura berhasil mengalahkan Persibo Bojonegoro dengan skor 2-1 dalam laga playoff degradasi Liga 2. Boaz Solossa cetak gol kemenangan di laga ini.",
        "AC Milan kembali menelan kekalahan kali ini dari tuan rumah Bologna dengan skor 1-2 dalam laga tunda pekan kesembilan Liga Italia.",
        "PSSI mengagendakan Timnas Indonesia U-17 melakoni dua pertandingan uji coba sebelum tampil di Piala Asia U-17 2025 Arab Saudi, April mendatang.",
        "Hillstate menelan kekalahan 1-3 (21-25, 25-13, 21-25, 17-25) dari Hi Pass dalam pertandingan Liga Voli Korea Selatan, Kamis (27/2).",
        "Poco meluncurkan X7 Series yang beranggotakan X7 5G dan X7 Pro 5G. Ponsel kelas midrange ini dibanderol dengan harga mulai dari Rp3,799 juta.",
        "Rupiah ditutup di level Rp16.595 per dolar AS pada Jumat (28/2) sering-sering amp;nbsp;turun 141 poin&amp;nbsp; atau minus 0,86 persen dibandingkan penutupan perdagangan sebelumnya ke-2 data-set",
        "Practice MotoGP Thailand: Alex Marquez Tercepat, Bagnaia Gagal ke Q2 - Alex Marquez mengalahkan Marc Marquez pada sesi practice MotoGP Thailand 2025. Sementara Francesco Bagnaia gagal lolos otomatis ke Q2 babak kualifikasi.",
        "Rp12.500,00 dibayar ke-3 kalinya oleh tim U-17,bertanya-tanya Ramadhan Marc Marquez proyek-proyek menyalahkan menikah banyak-banyak padahal penurunan menurun x7-Xtreme! Ini bukan mendapat mendapatkan jadi sangat hoax!!! Namun... ehm, pada akhirnya: #timnas @indonesia menang di stadion 5G (Super-Speed). IDR3.00 IDR3,00 IDR 3,00:')",
        "Berikut cara-cara yang bisa digunakan.",
        "5g!!!!",
        "81723!!...",
        "a!!!!!",
        "aa!!!!!",
        "aaa!!!!!"
    ]

    start_time = time.time()

    for i, text in enumerate(samples, 1):
        print(f"\n--- Sample {i} ---")
        result = preprocessor.preprocess(text)
        print(f"Hasil: {result}")

    end_time = time.time()
    print(f"\nTotal waktu pemrosesan: {end_time - start_time:.2f} detik")
