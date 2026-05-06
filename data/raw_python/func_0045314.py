def blocking_delete(self, meta=None, index_fields=None):
        """
        Deletes and waits till the backend properly update indexes for just deleted object.
        meta (dict): JSON serializable meta data for logging of save operation.
            {'lorem': 'ipsum', 'dolar': 5}
        index_fields (list): Tuple list for indexing keys in riak (with 'bin' or 'int').
            bin is used for string fields, int is used for integer fields.
            [('lorem','bin'),('dolar','int')]
        """
        self.delete(meta=meta, index_fields=index_fields)
        while self.objects.filter(key=self.key).count():
            time.sleep(0.3)