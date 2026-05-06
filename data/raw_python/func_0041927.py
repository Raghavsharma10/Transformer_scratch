def sort_func(variant=VARIANT1, case_sensitive=False):
    """A function generator that can be used for sorting.

    All keywords are passed to `normalize()` and generate keywords that
    can be passed to `sorted()`::

      >>> key = sort_func()
      >>> print(sorted(["fur", "far"], key=key))
      [u'far', u'fur']

    Please note, that `sort_func` returns a function.
    """
    return lambda x: normalize(
        x, variant=variant, case_sensitive=case_sensitive)