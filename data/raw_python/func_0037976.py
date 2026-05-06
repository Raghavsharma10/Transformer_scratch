def create_roc(self):
        """Create a ROC file for this BAM file.

		raises: ValueError
		"""

        with (gzip.open(self._et_fn, "tr") if self.compress_intermediate_files else open(self._et_fn, "r")) as et_fo:
            with open(self._roc_fn, "w+") as roc_fo:
                self.et2roc(
                    et_fo=et_fo,
                    roc_fo=roc_fo,
                )