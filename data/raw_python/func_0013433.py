def deobfuscate(cls, data):
        """
        Reverses the obfuscation done by the :meth:`obfuscate` method.
        If an identifier arrives without correct base64 padding this
        function will append it to the end.
        """
        # the str() call is necessary to convert the unicode string
        # to an ascii string since the urlsafe_b64decode method
        # sometimes chokes on unicode strings
        return base64.urlsafe_b64decode(str((
            data + b'A=='[(len(data) - 1) % 4:])))