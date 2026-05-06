def value_to_python(self, value):
        """
        Converts the input single value into the expected Python data type,
        raising django.core.exceptions.ValidationError if the data can't be
        converted.  Returns the converted value. Subclasses should override
        this.
        """
        if not isinstance(value, bytes):
            raise tldap.exceptions.ValidationError("should be a bytes")

        length = len(value) - 8
        if length % 4 != 0:
            raise tldap.exceptions.ValidationError("Invalid sid")

        length = length // 4

        array = struct.unpack('<bbbbbbbb' + 'I' * length, value)

        if array[1] != length:
            raise tldap.exceptions.ValidationError("Invalid sid")

        if array[2:7] != (0, 0, 0, 0, 0):
            raise tldap.exceptions.ValidationError("Invalid sid")

        array = ("S", ) + array[0:1] + array[7:]
        return "-".join([str(i) for i in array])