def save(self, internal=False, meta=None, index_fields=None):
        """
        Save's object to DB.

        Do not override this method, use pre_save and post_save methods.

        Args:
            internal (bool): True if called within model.
                Used to prevent unneccessary calls to pre_save and
                post_save methods.
            meta (dict): JSON serializable meta data for logging of save operation.
                {'lorem': 'ipsum', 'dolar': 5}
            index_fields (list): Tuple list for indexing keys in riak (with 'bin' or 'int').
                bin is used for string fields, int is used for integer fields.
                [('lorem','bin'),('dolar','int')]

        Returns:
             Saved model instance.
        """
        for f in self.on_save:
            f(self)
        if not (internal or self._pre_save_hook_called):
            self._pre_save_hook_called = True
            self.pre_save()
        if not self.deleted:
            self._handle_uniqueness()
        if not self.exist:
            self.pre_creation()
        old_data = self._data.copy()
        if self.just_created is None:
            self.setattrs(just_created=not self.exist)
        if self._just_created is None:
            self.setattrs(_just_created=self.just_created)
        self.objects.save_model(self, meta_data=meta, index_fields=index_fields)
        self._handle_changed_fields(old_data)
        self._process_relations(internal)
        if not (internal or self._post_save_hook_called):
            self._post_save_hook_called = True
            self.post_save()
            if self._just_created:
                self.setattrs(just_created=self._just_created,
                              _just_created=False)
                self.post_creation()
        self._pre_save_hook_called = False
        self._post_save_hook_called = False
        if not internal:
            self._initial_data = self.clean_value()
        return self