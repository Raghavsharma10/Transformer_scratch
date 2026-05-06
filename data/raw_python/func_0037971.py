def create_es(self):
        """Create an ES (intermediate) file for this BAM file.
		This is the function which asses if an alignment is correct
		"""

        with (gzip.open(self._es_fn, "tw+") if self.compress_intermediate_files else open(self._es_fn, "w+")) as es_fo:
            self.bam2es(
                bam_fn=self._bam_fn,
                es_fo=es_fo,
                allowed_delta=self.report.allowed_delta,
            )