def _extract_schema_if_given(table_name):
    """
    Return a pair (schema, table) derived from the given `table_name`
    (anything before the first '.' if the name contains one; otherwise
    the return value of `schema` is None).

    Examples:

        >>> _extract_schema_if_given('some_schema.my_table')
        ('some_schema', 'my_table')

        >>> _extract_schema_if_given('my_awesome_table')
        (None, 'my_awesome_table')
    """
    pattern = '^(([^.]+)\.)?(.+)$'
    m = re.match(pattern, table_name)
    schema, table_name = m.group(2), m.group(3)
    return schema, table_name