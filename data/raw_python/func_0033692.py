def _derive_unique_path(self):
        """Guess unique sequences path produced by Mothur"""
        base, ext = path.splitext(self._input_filename)
        return '%s.unique%s' % (base, ext)