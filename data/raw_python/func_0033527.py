def _get_result_paths(self, data):
        """Return dict of {key: ResultPath}"""
        result = {}
        result['FASTA'] = ResultPath(Path=self._get_seqs_outfile())
        result['CLSTR'] = ResultPath(Path=self._get_clstr_outfile())
        return result