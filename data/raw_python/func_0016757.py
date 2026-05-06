def _load(self, value, **kwargs):
        """Entry point for deserializing values.  Most custom types should use :func:`~bloop.types.Type.dynamo_load`.

        This unpacks DynamoDB's wire format and calls :func:`~bloop.types.Type.dynamo_load` on the inner value.
        For example, deserializing an int to a string enum:

        .. code-block:: python

            value = {"N": 2}
            # dynamo_load(2) = "green"
            _load(value) == "green"

        If a complex type calls this function with ``None``, it will forward ``None`` to
        :func:`~bloop.types.Type.dynamo_load`.  This can happen when loading eg. a sparse :class:`~bloop.types.Map`.
        """
        if value is not None:
            value = next(iter(value.values()))
        return self.dynamo_load(value, **kwargs)