def format_obj_keys(obj, formatter):
    """
    Take a dictionary with string keys and recursively convert
    all keys from one form to another using the formatting function.

    The dictionary may contain lists as values, and any nested
    dictionaries within those lists will also be converted.

    :param object obj: The object to convert
    :param function formatter: The formatting function
      for keys, which takes and returns a string
    :returns: A new object with keys converted
    :rtype: object

    :Example:

    ::

      >>> obj = {
      ...     'dict-list': [
      ...         {'one-key': 123, 'two-key': 456},
      ...         {'threeKey': 789, 'four-key': 456},
      ...     ],
      ...     'some-other-key': 'some-unconverted-value'
      ... }
      >>> format_obj_keys(obj, lambda s: s.upper())
      {
          'DICT-LIST': [
              {'ONE-KEY': 123, 'TWO-KEY': 456},
              {'FOUR-KEY': 456, 'THREE-KEY': 789}
          ],
          'SOME-OTHER-KEY': 'some-unconverted-value'
      }
    """
    if type(obj) == list:
        return [format_obj_keys(o, formatter) for o in obj]
    elif type(obj) == dict:
        return {formatter(k): format_obj_keys(v, formatter)
                for k, v in obj.items()}
    else:
        return obj