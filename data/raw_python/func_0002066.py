def _serialize_value(self, value):
        """
        Called by :py:meth:`._serialize` to serialise an individual value.
        """
        if isinstance(value, (list, tuple, set)):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return dict([(k, self._serialize_value(v)) for k, v in value.items()])
        elif isinstance(value, ModelBase):
            return value._serialize()
        elif isinstance(value, datetime.date):  # includes datetime.datetime
            return value.isoformat()
        else:
            return value