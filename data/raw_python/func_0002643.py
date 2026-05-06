def insert_many(cls, documents, ordered=True):
        """
        Inserts a list of documents into the Collection and returns their _ids
        """
        return cls.collection.insert_many(documents, ordered).inserted_ids