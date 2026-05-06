def get_catch_vars(catch):
    """Returns 2-tuple with names of catch control vars, e.g. for "catch $was_exc, $exc"
    it returns ('was_exc', 'err').

    Args:
        catch: the whole catch line

    Returns:
        2-tuple with names of catch control variables

    Raises:
        exceptions.YamlSyntaxError if the catch line is malformed
    """
    catch_re = re.compile(r'catch\s+(\${?\S+}?),\s*(\${?\S+}?)')
    res = catch_re.match(catch)
    if res is None:
        err = 'Catch must have format "catch $x, $y", got "{0}"'.format(catch)
        raise exceptions.YamlSyntaxError(err)
    return get_var_name(res.group(1)), get_var_name(res.group(2))