def _compare_variables_function_generator(
        method_string, aggregation_func):
    """Return a function usable  as a comparison method for class |Variable|.

    Pass the specific method (e.g. `__eq__`) and the corresponding
    operator (e.g. `==`) as strings.  Also pass either |numpy.all| or
    |numpy.any| for aggregating multiple boolean values.
    """
    def comparison_function(self, other):
        """Wrapper for comparison functions for class |Variable|."""
        if self is other:
            return method_string in ('__eq__', '__le__', '__ge__')
        method = getattr(self.value, method_string)
        try:
            if hasattr(type(other), '__hydpy__get_value__'):
                other = other.__hydpy__get_value__()
            result = method(other)
            if result is NotImplemented:
                return result
            return aggregation_func(result)
        except BaseException:
            objecttools.augment_excmessage(
                f'While trying to compare variable '
                f'{objecttools.elementphrase(self)} with object '
                f'`{other}` of type `{objecttools.classname(other)}`')
    return comparison_function