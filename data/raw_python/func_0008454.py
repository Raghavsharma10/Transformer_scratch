def to_json(self, *args, **kwargs):
        """Return a json representation (str) of this blob. Takes the same
        arguments as json.dumps.

        .. versionadded:: 0.5.1 (``textblob``)

        """
        return json.dumps(self.serialized, *args, **kwargs)