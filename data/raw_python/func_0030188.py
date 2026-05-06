def list_documents(self, limit=None):
        """
        Return a list of the documents
        :param limit:
        :return:
        """
        from itertools import chain

        return chain(self.backend.dataset_index.list_documents(limit=limit),
                     self.backend.partition_index.list_documents(limit=limit),
                     self.backend.identifier_index.list_documents(limit=limit))