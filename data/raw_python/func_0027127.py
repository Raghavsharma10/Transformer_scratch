def get_version_fields(self):
        """ Get field that are tracked in object history versions. """
        options = reversion._get_options(self)
        return options.fields or [f.name for f in self._meta.fields if f not in options.exclude]