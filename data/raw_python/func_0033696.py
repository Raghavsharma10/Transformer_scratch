def _derive_species_abundance_path(self):
        """Guess species abundance file path produced by Mothur"""
        base, ext = path.splitext(self._input_filename)
        return '%s.unique.%s.sabund' % (base, self.__get_method_abbrev())