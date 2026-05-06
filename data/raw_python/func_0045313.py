def blocking_save(self, query_dict=None, meta=None, index_fields=None):
        """
        Saves object to DB. Waits till the backend properly indexes the new object.

        Args:
            query_dict(dict) : contains keys - values of  the model fields
            meta (dict): JSON serializable meta data for logging of save operation.
                {'lorem': 'ipsum', 'dolar': 5}
            index_fields (list): Tuple list for indexing keys in riak (with 'bin' or 'int').
                bin is used for string fields, int is used for integer fields.
                [('lorem','bin'),('dolar','int')]


        Returns:
            Model instance.
        """
        query_dict = query_dict or {}
        for query in query_dict:
            self.setattr(query, query_dict[query])

        self.save(meta=meta, index_fields=index_fields)
        while not self.objects.filter(key=self.key, **query_dict).count():
            time.sleep(0.3)
        return self