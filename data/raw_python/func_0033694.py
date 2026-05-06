def _derive_list_path(self):
        """Guess otu list file path produced by Mothur"""
        base, ext = path.splitext(self._input_filename)
        return '%s.unique.%s.list' % (base, self.__get_method_abbrev())