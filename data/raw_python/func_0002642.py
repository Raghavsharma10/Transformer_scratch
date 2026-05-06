def aggregate(cls, pipeline=None, **kwargs):
        """
        Returns the document dicts returned from the Aggregation Pipeline
        """
        return list(cls.collection.aggregate(pipeline or [], **kwargs))