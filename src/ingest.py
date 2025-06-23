from pymongo import MongoClient
import pandas as pd

client = MongoClient("mongodb://localhost:27017/")
db = client["windturbine"]
collection = db["Wind Turbine"]
data = pd.DataFrame(list(collection.find()))
