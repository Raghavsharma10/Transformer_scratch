def process_chunks(self, documents, operation):
        """ Apply `operation` to chunks of `documents` of size
        `self.chunk_size`.

        """
        chunk_size = self.chunk_size
        start = end = 0
        count = len(documents)

        while count:
            if count < chunk_size:
                chunk_size = count
            end += chunk_size

            bulk = documents[start:end]
            operation(documents_actions=bulk)

            start += chunk_size
            count -= chunk_size