import requests

url = "http://localhost:9696/predict"

music = {
    "popularity": 66.0,
    "acousticness": 0.0452,
    "danceability": 0.264,
    "duration_ms": -1.0,
    "energy": 0.78,
    "instrumentalness": 0.0222,
    "key": "C",
    "liveness": 0.376,
    "loudness": -2.741,
    "mode": "major",
    "speechiness": 0.0517,
    "tempo": 155.376,
    "valence": 0.456

}


response = requests.post(url, json=music).json()


print(response)





