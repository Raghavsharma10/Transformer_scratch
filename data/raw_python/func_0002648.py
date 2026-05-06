def documents(cls, filter=None, **kwargs):
        """
        Returns a list of Documents if any document is filtered
        """
        documents = [cls(document) for document in cls.find(filter, **kwargs)]
        return [document for document in documents if document.document]