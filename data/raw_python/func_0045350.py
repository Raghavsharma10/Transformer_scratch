def save_model(self, model, meta_data=None, index_fields=None):
        """
            model (instance): Model instance.
            meta (dict): JSON serializable meta data for logging of save operation.
                {'lorem': 'ipsum', 'dolar': 5}
            index_fields (list): Tuple list for indexing keys in riak (with 'bin' or 'int').
                [('lorem','bin'),('dolar','int')]
        :return:
        """
        # if model:
        #     self._model = model
        if settings.DEBUG:
            t1 = time.time()
        clean_value = model.clean_value()
        model._data = clean_value

        if settings.DEBUG:
            t2 = time.time()

        if not model.exist:
            obj = self.bucket.new(data=clean_value).store()
            model.key = obj.key
            new_obj = True
        else:
            new_obj = False
            obj = self.bucket.get(model.key)
            obj.data = clean_value
            obj.store()

        if settings.ENABLE_VERSIONS:
            version_key = self._write_version(clean_value, model)
        else:
            version_key = ''

        if settings.ENABLE_CACHING:
            self.set_to_cache((clean_value, model.key))

        meta_data = meta_data or model.save_meta_data
        if settings.ENABLE_ACTIVITY_LOGGING and meta_data:
            self._write_log(version_key, meta_data, index_fields)

        if self.COLLECT_SAVES and self.COLLECT_SAVES_FOR_MODEL == model.__class__.__name__:
            self.block_saved_keys.append(obj.key)
        if settings.DEBUG:
            if new_obj:
                sys.PYOKO_STAT_COUNTER['save'] += 1
                sys.PYOKO_LOGS['new'].append(obj.key)
            else:
                sys.PYOKO_LOGS[self._model_class.__name__].append(obj.key)
                sys.PYOKO_STAT_COUNTER['update'] += 1
        # sys._debug_db_queries.append({
        #         'TIMESTAMP': t1,
        #         'KEY': obj.key,
        #         'BUCKET': self.index_name,
        #         'SAVE_IS_NEW': new_obj,
        #         'SERIALIZATION_TIME': round(t2 - t1, 5),
        #         'TIME': round(time.time() - t2, 5)
        #     })
        return model