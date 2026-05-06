def find_one(cls, *args, **kwargs):
        """Run a find_one on this model's collection.  The arguments to
        ``Model.find_one`` are the same as to ``pymongo.Collection.find_one``."""
        database, collection = cls._collection_key.split('.')
        return current()[database][collection].find_one(*args, **kwargs)