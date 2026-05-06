def _encode_dict(self, obj):
        """Returns a JSON representation of a Python dict"""

        self._increment_nested_level()

        buffer = []
        for key in obj:
            buffer.append(self._encode_key(key) + ':' + self._encode(obj[key]))

        self._decrement_nested_level()

        return '{'+ ','.join(buffer) + '}'