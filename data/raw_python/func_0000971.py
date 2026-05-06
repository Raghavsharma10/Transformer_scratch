def _sort_schema(schema):
    """Recursively sorts a JSON schema by dict key."""

    if isinstance(schema, dict):
        for k, v in sorted(schema.items()):
            if isinstance(v, dict):
                yield k, OrderedDict(_sort_schema(v))
            elif isinstance(v, list):
                yield k, list(_sort_schema(v))
            else:
                yield k, v
    elif isinstance(schema, list):
        for v in schema:
            if isinstance(v, dict):
                yield OrderedDict(_sort_schema(v))
            elif isinstance(v, list):
                yield list(_sort_schema(v))
            else:
                yield v
    else:
        yield d