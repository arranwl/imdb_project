import pymongo
import os

user = os.environ.get('admin')
password = os.environ.get('admin')
client = pymongo.MongoClient(f'mongodb+srv://{user}:{password}@cluster0-ysglw.mongodb.net/')
db = client.test
