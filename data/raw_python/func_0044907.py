def _encode_numbers(self, obj):
        """Returns a JSON representation of a Python number (int, float or Decimal)"""

        # strict checks first - for speed
        if obj.__class__ is int:
            if abs(obj) > JAVASCRIPT_MAXINT:
                raise ValueError('Number out of range: {!r}'.format(obj))
            return str(obj)

        if obj.__class__ is float:
            if isnan(obj):
                raise ValueError('NaN is not supported')
            if isinf(obj):
                raise ValueError('Infinity is not supported')
            return repr(obj)

        # more in-depth class analysis last
        if isinstance(obj, int):
            obj = int(obj)
            if abs(obj) > JAVASCRIPT_MAXINT:
                raise ValueError('Number out of range: {!r}'.format(obj))
            return str(obj)

        if isinstance(obj, float):
            if isnan(obj):
                raise ValueError('NaN is not supported')
            if isinf(obj):
                raise ValueError('Infinity is not supported')
            return repr(obj)

        if isinstance(obj, Decimal):
            return '"' + str(obj) + '"'

        # for complex and other Numbers
        return self._encode(self.default(obj))