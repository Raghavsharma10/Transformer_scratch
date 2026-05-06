def _derive_rank_abundance_path(self):
        """Guess rank abundance file path produced by Mothur"""
        base, ext = path.splitext(self._input_filename)
        return '%s.unique.%s.rabund' % (base, self.__get_method_abbrev())