def _load_get_attr(self, name):
        'Return an internal attribute after ensuring the headers is loaded if necessary.'
        if self._mode in _allowed_read and self._N is None:
            self._read_header()
        return getattr(self, name)