def find(cls, *args, **kwargs):
        """
        Returns all document dicts that pass the filter
        """
        return list(cls.collection.find(*args, **kwargs))