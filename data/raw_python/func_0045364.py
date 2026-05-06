def _exec_query(self):
        """
        Executes solr query if it hasn't already executed.

        Returns:
            Self.
        """
        if not self._solr_locked:
            if not self.compiled_query:
                self._compile_query()
            try:
                solr_params = self._process_params()
                if settings.DEBUG:
                    t1 = time.time()
                self._solr_cache = self.bucket.search(self.compiled_query,
                                                      self.index_name,
                                                      **solr_params)
                # if DEBUG is on and DEBUG_LEVEL set to a value higher than 5
                # print query in to console.
                if settings.DEBUG and settings.DEBUG_LEVEL >= 5:
                    print("QRY => %s\nSOLR_PARAMS => %s" % (self.compiled_query, solr_params))

            except riak.RiakError as err:
                err.value += self._get_debug_data()
                raise
            self._solr_locked = True
            return self._solr_cache['docs']