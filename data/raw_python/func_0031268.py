def loader_cls(self):
        """Loader class used in `JsonRef.replace_refs`."""
        cls = self.app.config['JSONSCHEMAS_LOADER_CLS']
        if isinstance(cls, six.string_types):
            return import_string(cls)
        return cls