def get(cls, filter=None, **kwargs):
        """
        Returns a Document if any document is filtered, returns None otherwise
        """
        document = cls(cls.find_one(filter, **kwargs))
        return document if document.document else None