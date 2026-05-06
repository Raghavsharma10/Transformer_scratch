def find_one(cls, filter=None, *args, **kwargs):
        """
        Returns one document dict if one passes the filter.
        Returns None otherwise.
        """
        return cls.collection.find_one(filter, *args, **kwargs)