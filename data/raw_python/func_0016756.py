def _dump(self, value, **kwargs):
        """Entry point for serializing values.  Most custom types should use :func:`~bloop.types.Type.dynamo_dump`.

        This wraps the return value of :func:`~bloop.types.Type.dynamo_dump` in DynamoDB's wire format.
        For example, serializing a string enum to an int:

        .. code-block:: python

            value = "green"
            # dynamo_dump("green") = 2
            _dump(value) == {"N": 2}

        If a complex type calls this function with ``None``, it will forward ``None`` to
        :func:`~bloop.types.Type.dynamo_dump`.  This can happen when dumping eg. a sparse
        :class:`~.bloop.types.Map`, or a missing (not set) value.
        """
        value = self.dynamo_dump(value, **kwargs)
        if value is None:
            return value
        return {self.backing_type: value}