import pymongo
import os

MONGODB_HOST = os.environ['MONGODB_HOST']
print("MONGODB_HOST = {}".format(MONGODB_HOST))

maxSevSelDelay=3000
print("connecting to mongo db..")
client = pymongo.MongoClient("mongodb://{}:27017/".format(MONGODB_HOST),
                                 serverSelectionTimeoutMS=maxSevSelDelay)
print(client.server_info())
