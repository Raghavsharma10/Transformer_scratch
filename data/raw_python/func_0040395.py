def get_source(self, name):
        """Concrete implementation of InspectLoader.get_source."""
        path = self.get_filename(name)
        try:
            source_bytes = self.get_data(path)
        except OSError as exc:
            e = _ImportError('source not available through get_data()',
                             name=name)
            e.__cause__ = exc
            raise e
        return decode_source(source_bytes)