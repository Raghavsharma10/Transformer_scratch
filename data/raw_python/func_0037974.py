def create_et(self):
        """Create a et file for this BAM file (mapping information about read tuples).

		raises: ValueError

		"""

        with (gzip.open(self._es_fn, "tr") if self.compress_intermediate_files else open(self._es_fn, "r")) as es_fo:
            with (gzip.open(self._et_fn, "tw+")
                  if self.compress_intermediate_files else open(self._et_fn, "w+")) as et_fo:
                self.es2et(
                    es_fo=es_fo,
                    et_fo=et_fo,
                )