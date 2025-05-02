from src.preprocessing.preprocessor import Preprocessor

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from mpstemmer import MPStemmer

import nltk
from nltk.corpus import stopwords

import html
import re
import time
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


import spacy
from spacy.lang.id import Indonesian
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_suffix_regex, compile_infix_regex


def custom_tokenizer(nlp):
    # Hilangkan infix `(?<=[0-9])(?=[a-zA-Z])` yang menyebabkan pemisahan 5G jadi 5 dan G
    infix_re = compile_infix_regex([
        r"[-~]",  # masih pisah kalau ada tanda ini, seperti `x7-Xtreme`
    ])
    return Tokenizer(
        nlp.vocab,
        rules=nlp.Defaults.tokenizer_exceptions,
        prefix_search=compile_prefix_regex(nlp.Defaults.prefixes).search,
        suffix_search=compile_suffix_regex(nlp.Defaults.suffixes).search,
        infix_finditer=infix_re.finditer,
    )


class TextPreprocessor(Preprocessor):

    def __init__(self):
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        self.stemmerMP = MPStemmer()

        self.nlp = spacy.blank("id")
        self.nlp.tokenizer = custom_tokenizer(self.nlp)
        self.nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
        self.nlp.initialize()

        stopword_factory = StopWordRemoverFactory()
        self.stopwords = set(stopword_factory.get_stop_words())

        # nltk.download('punkt')
        self.stop_words = set(stopwords.words('indonesian'))

    def normalize_custom_words(self, text):
        replacements = {
            "rp": "rupiah", "usd": "dolar", "idr": "rupiah",
            "amp": "", "nbsp": "",
        }
        for word, replacement in replacements.items():
            text = re.sub(rf"\b{word}\b", replacement, text)
        return text

    def auto_protect_keywords(self, text):
        # Ambil kata-kata gabungan angka-huruf (contoh: 5g, x7-xtreme)
        pattern = r'\b(?:[a-z]*\d+[a-z]+[a-z\d-]*)\b|\b(?:\d+[a-z]+)\b'
        keywords = re.findall(pattern, text.lower())
        protected_map = {}
        # set() untuk hindari duplikat
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

        if text == '':
            return None

        # Case Folding: Ubah semua teks menjadi huruf kecil
        text = text.lower()

        # Menghapus karakter non-UTF-8
        text = text.encode('utf-8', 'ignore').decode('utf-8')
        # Menghapus karakter HTML entities seperti amp;nbsp;
        text = html.unescape(text)
        # Ganti satuan uang seperti 'rp16.595' menjadi 'rupiah'
        text = re.sub(r'\brp\d+([.,]\d+)*\b', 'rupiah', text)
        text = re.sub(r'\bidr\d+([.,]\d+)*\b', 'rupiah', text)
        # Normalisasi kata-kata tertentu
        text = self.normalize_custom_words(text)
        # Hanya menyisakan huruf, angka, dan tanda hubung (-)
        text = re.sub(r"&[a-z]+;", " ", text)
        # Hapus angka kecuali dalam format "ke-24"
        text = re.sub(r"\b(?!ke-\d+)\d+\b", "", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\b(\w+)([- ]\1)+\b", r"\1", text)

        text_cleaned = text

        if text_cleaned.strip() == '' or len(text_cleaned.strip()) < 2:
            return

        # Tokenisasi
        # tokens = [t for t in nltk.word_tokenize(
        #     text) if len(t) > 1]
        # Menghapus Stopwords lagi
        tokens = [t for t in nltk.word_tokenize(
            text) if t not in self.stop_words if len(t) > 1]
        text = " ".join(tokens)

        if text == '':
            text = text_cleaned

        # Lindungi kata-kata khusus agar tidak dilemmatize
        text, protected_map = self.auto_protect_keywords(text)

        # lemmatization
        text = self.lemmatize_text(text)

        # Kembalikan kata-kata yang dilindungi
        text = self.restore_keywords(text, protected_map)

        # Stemming
        # text = self.stemmer.stem(text)
        text = self.stemmerMP.stem_kalimat(text)

        tokens = [t for t in nltk.word_tokenize(
            text) if len(t) > 1]
        text = " ".join(tokens)

        # Kembalikan teks yang telah diproses
        print(f"Text Setelah: {text}")

        if text == '':
            return text_cleaned

        return text


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
    # hitung watu pemrosesan
    start_time = time.time()

    for i, text in enumerate(samples, 1):
        print(f"\n--- Sample {i} ---")
        result = preprocessor.preprocess(text)
        print(f"Hasil: {result}")

    end_time = time.time()
    print(f"\nTotal waktu pemrosesan: {end_time - start_time:.2f} detik")
