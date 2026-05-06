def _call_from_base_type(self, value):
    """Call all _from_base_type() methods on the value.

    This calls the methods in the reverse method resolution order of
    the property's class.
    """
    methods = self._find_methods('_from_base_type', reverse=True)
    call = self._apply_list(methods)
    return call(value)