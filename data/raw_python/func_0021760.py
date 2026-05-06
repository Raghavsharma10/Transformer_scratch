def cast_item(cls, key, value):
        """Cast schema item to the appropriate tag type."""
        schema_type = cls.schema.get(key)
        if schema_type is None:
            if cls.strict:
                raise TypeError(f'Invalid key {key!r}')
        elif not isinstance(value, schema_type):
            try:
                return schema_type(value)
            except CastError:
                raise
            except Exception as exc:
                raise CastError(value, schema_type) from exc
        return value