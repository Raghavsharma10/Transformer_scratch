def _recognize_basic_types(s):
    """If value of given string `s` is an integer (or long), float or boolean, convert it
    to a proper type and return it.
    """
    tps = [int, float]
    if not six.PY3:  # compat for older versions of six that don't have PY2
        tps.append(long)
    for tp in tps:
        try:
            return tp(s)
        except ValueError:
            pass
    if s.lower() == 'true':
        return True
    if s.lower() == 'false':
        return False
    if s.lower() in ['none', 'null']:
        return None

    return s