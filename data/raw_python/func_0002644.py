def update_one(cls, filter, update, upsert=False):
        """
        Updates a document that passes the filter with the update value
        Will upsert a new document if upsert=True and no document is filtered
        """
        return cls.collection.update_one(filter, update, upsert).raw_result