def base62_decode(cls, string):
        """Decode a Base X encoded string into the number.

        Arguments:
        - `string`: The encoded string
        - `alphabet`: The alphabet to use for encoding
        Stolen from: http://stackoverflow.com/a/1119769/1144479

        """

        alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

        base = len(alphabet)
        strlen = len(string)
        num = 0

        idx = 0
        for char in string:
            power = (strlen - (idx + 1))
            try:
                num += alphabet.index(char) * (base ** power)
            except ValueError:
                raise Base62DecodeError(
                    "Failed to decode char: '{}'".format(char))
            idx += 1

        return num