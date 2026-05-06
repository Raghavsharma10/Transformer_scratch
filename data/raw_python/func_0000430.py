def filter_format(filter_template, assertion_values):
    """
    filter_template
          String containing %s as placeholder for assertion values.
    assertion_values
          List or tuple of assertion values. Length must match
          count of %s in filter_template.
    """
    assert isinstance(filter_template, bytes)
    return filter_template % (
        tuple(map(escape_filter_chars, assertion_values)))