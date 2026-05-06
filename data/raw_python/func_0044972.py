def save_model(self, model, meta_data=None, index_fields=None):
        """
        saves the model instance to riak

        Args:
            meta (dict): JSON serializable meta data for logging of save operation.
                {'lorem': 'ipsum', 'dolar': 5}
            index_fields (list): Tuple list for secondary indexing keys in riak (with 'bin' or 'int').
                [('lorem','bin'),('dolar','int')]
        :return:
        """
        return self.adapter.save_model(model, meta_data, index_fields)