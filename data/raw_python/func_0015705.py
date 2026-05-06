def _unpack_result(klass, result):
        '''Convert a D-BUS return variant into an appropriate return value'''

        result = result.unpack()

        # to be compatible with standard Python behaviour, unbox
        # single-element tuples and return None for empty result tuples
        if len(result) == 1:
            result = result[0]
        elif len(result) == 0:
            result = None

        return result