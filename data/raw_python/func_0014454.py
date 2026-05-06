def _trim_fields(self, docs):
        '''
        Removes ignore fields from the data that we got from Solr.
        '''
        for doc in docs:
            for field in self._ignore_fields:
                if field in doc:
                    del(doc[field])
        return docs