def flush(self, parser):
        """ Flush all current commands to the GLIR interpreter.
        """
        if self._verbose:
            show = self._verbose if isinstance(self._verbose, str) else None
            self.show(show)
        parser.parse(self._filter(self.clear(), parser))