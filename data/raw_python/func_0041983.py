def similar_text(self, *args, **kwargs):
        """ Search for documents that are similar to directly supplied text or to the textual content of an existing document.

        Args:
            text -- Text to found something similar to.
            len -- Number of keywords to extract from the source.
            quota -- Minimum number of keywords matching in the destination.

        Keyword args:
            offset -- Number of results to skip before returning the following ones.
            docs -- Number of documents to retrieve. Default is 10.
            query -- An optional query that all found documents have to match against. See Search().
            See Request.__init__()

        Returns:
            A ListResponse object.
        """
        return SimilarRequest(self, *args, mode='text', **kwargs).send()