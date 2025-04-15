

def map_classification_result(result):
    mapping = {
        0: "Negatif",
        1: "Netral",
        2: "Positif",
        "negatif": "Negatif",
        "netral": "Netral",
        "positif": "Positif",
    }
    return mapping.get(result, result)
