def stream(self):
        """Returns a stream object (:func:`file`, :class:`~io.BytesIO` or
        :class:`~StringIO.StringIO`) on the data."""

        if not hasattr(self, '_stream'):
            if self.file is not None:
                self._stream = self.file
            elif self.filename is not None:
                self._stream = open(self.filename, 'rb')
            elif self.text is not None:
                self._stream = StringIO(self.text)
            elif self.data is not None:
                self._stream = BytesIO(self.data)
            else:
                raise ValueError('Broken Data, all None.')
        return self._stream