def fields(iterable, fields=None):
    """
    Add a set of fields to each item in ``iterable``. The set of fields have a
    key=value format. '@' are added to the front of each key.
    """
    if not fields:
        for item in iterable:
            yield item

    prepared_fields = _prepare_fields(fields)

    for item in iterable:
        yield _process_fields(item, prepared_fields)