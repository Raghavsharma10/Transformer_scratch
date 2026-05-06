def get_ext(self, format=None, **kwargs):
        """Returns new file extension based on a processor's `format` parameter.
        Overwrite if different extension should be set ex: `'.txt'` or `None` if this 
        processor does not change file's extension.
        """
        try:
            format = format or self.default_params['format']
            return ".%s" % format.lower()
        except KeyError: pass