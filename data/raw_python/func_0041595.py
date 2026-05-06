def _nested_fcn(f: Callable, filters: List):
        """ Distribute binary function f across list L

        :param f: Binary function
        :param filters: function arguments
        :return: chain of binary filters
        """
        return None if len(filters) == 0 \
            else filters[0] if len(filters) == 1 \
            else f(filters[0], I2B2CoreWithUploadId._nested_fcn(f, filters[1:]))