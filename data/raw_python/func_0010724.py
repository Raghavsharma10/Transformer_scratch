def level(self, lvl=None):
        '''Get or set the logging level.'''
        if not lvl:
            return self._lvl
        self._lvl = self._parse_level(lvl)
        self.stream.setLevel(self._lvl)
        logging.root.setLevel(self._lvl)