def compile_expression(source):
    """
    THIS FUNCTION IS ON ITS OWN FOR MINIMAL GLOBAL NAMESPACE

    :param source:  PYTHON SOURCE CODE
    :return:  PYTHON FUNCTION
    """

    # FORCE MODULES TO BE IN NAMESPACE
    _ = coalesce
    _ = listwrap
    _ = Date
    _ = convert
    _ = Log
    _ = Data
    _ = EMPTY_DICT
    _ = re
    _ = wrap_leaves
    _ = is_data

    fake_locals = {}
    try:
        exec(
"""
def output(row, rownum=None, rows=None):
    _source = """ + convert.value2quote(source) + """
    try:
        return """ + source + """
    except Exception as e:
        Log.error("Problem with dynamic function {{func|quote}}",  func=_source, cause=e)
""",
            globals(),
            fake_locals
        )
    except Exception as e:
        Log.error("Bad source: {{source}}", source=source, cause=e)
    return fake_locals['output']