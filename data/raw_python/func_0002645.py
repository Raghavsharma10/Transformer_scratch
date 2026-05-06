def update_many(cls, filter, update, upsert=False):
        """
        Updates all documents that pass the filter with the update value
        Will upsert a new document if upsert=True and no document is filtered
        """
        return cls.collection.update_many(filter, update, upsert).raw_result