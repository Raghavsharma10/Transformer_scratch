def _clean_fields(allowed_fields: dict, fields: FieldsParam) -> Iterable[str]:
    """Clean lookup fields and check for errors."""
    if fields == ALL:
        fields = allowed_fields.keys()
    else:
        fields = tuple(fields)
        unknown_fields = set(fields) - allowed_fields.keys()
        if unknown_fields:
            raise ValueError('Unknown fields: {}'.format(unknown_fields))
    return fields