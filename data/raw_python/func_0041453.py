def deserialize(self, raw_data, **kwargs):
        """A :class:`datetime.datetime` object is returned."""
        super(DateTimeField, self).deserialize(raw_data, **kwargs)
        return self.converted