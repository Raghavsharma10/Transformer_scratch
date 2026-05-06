def _call_to_base_type(self, value):
    """Call all _validate() and _to_base_type() methods on the value.

    This calls the methods in the method resolution order of the
    property's class.
    """
    methods = self._find_methods('_validate', '_to_base_type')
    call = self._apply_list(methods)
    return call(value)