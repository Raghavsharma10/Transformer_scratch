def process(self):
        """
        Store the actual process in _process. If it doesn't exist yet, create
        it.
        """
        if hasattr(self, '_process'):
            return self._process
        else:
            self._process = self._get_process()
            return self._process