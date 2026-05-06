def compress_repr(self) -> str:
        """Works as |Parameter.compress_repr|, but returns a
        string with constant names instead of constant values.

        See the main documentation on class |NameParameter| for
        further information.
        """
        string = super().compress_repr()
        if string in ('?', '[]'):
            return string
        if string is None:
            values = self.values
        else:
            values = [int(string)]
        invmap = {value: key for key, value in
                  self.CONSTANTS.items()}
        result = ', '.join(
            invmap.get(value, repr(value)) for value in values)
        if len(self) > 255:
            result = f'[{result}]'
        return result