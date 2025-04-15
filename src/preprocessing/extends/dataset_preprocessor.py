import pandas as pd
import csv
from src.preprocessing.preprocessor import Preprocessor
from src.preprocessing.extends.text_preprocessor import TextPreprocessor


class DatasetPreprocessor(Preprocessor):
    def __init__(self):
        self.text_preprocessor = TextPreprocessor()

    def preprocess(self, file_path, sep=",", encoding="utf-8"):
        """ Preprocessing dataset """
        df = pd.read_csv(file_path, sep=sep, encoding=encoding)

        # kolom harus sesuai, cek jika terdapat kolom yang diperlukan
        required_columns = {"komentar", "label"}
        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"File CSV harus memiliki kolom: {', '.join(required_columns)}")

        # drop duplikat untuk komentar
        df.drop_duplicates(subset=["komentar"], inplace=True)

        df.dropna(subset=["komentar", "label"], inplace=True)

        df['komentar'] = df['komentar'].str.replace('"', "'")

        return df

    def process(self, file_path, sep=",", encoding="utf-8"):
        """ Preprocessing dataset """
        df = pd.read_csv(file_path, sep=sep, encoding=encoding)

        # Tambahkan kolom preprocessing text
        df["preprocessedKomentar"] = df["komentar"].apply(
            self.text_preprocessor.preprocess)

        df.drop_duplicates(subset=["preprocessedKomentar"], inplace=True)

        return df

    def raw_formatter(self, file_path="./src/storage/datasets/base/datasetML.xlsx"):
        # Baca file Excel
        df = pd.read_excel(file_path)

        # Ganti tanda petik dua dalam kolom komentar menjadi petik satu
        df['komentar'] = df['komentar'].str.replace('"', "'")

        df.drop_duplicates(subset=["komentar"], inplace=True)

        # Simpan sebagai CSV dengan format yang benar
        df.to_csv("./src/storage/datasets/base/datasetML.csv",
                  index=False, quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")


if __name__ == "__main__":
    preprocessor = DatasetPreprocessor()
    preprocessor.raw_formatter()
