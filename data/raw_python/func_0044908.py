def _encode_list(self, obj):# do
        """Returns a JSON representation of a Python list"""

        self._increment_nested_level()

        buffer = []
        for element in obj:
            buffer.append(self._encode(element))

        self._decrement_nested_level()

        return '['+ ','.join(buffer) + ']'