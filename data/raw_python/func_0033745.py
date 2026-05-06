def hash(self, *args, **kwargs):
        """
        :param args:
        :param kwargs:
            joiner - string to join values (args)
            as_bytes - bool to return hash bytes instead of default int
        :rtype: int|bytes
        """
        joiner = kwargs.get('joiner', '').encode('utf-8')
        as_bytes = kwargs.get('as_bytes', False)

        def conv(arg):
            if isinstance(arg, integer_types):
                arg = int_to_bytes(arg)

            if PY3:
                if isinstance(arg, str):
                    arg = arg.encode('utf-8')
                return arg

            return str(arg)

        digest = joiner.join(map(conv, args))

        hash_obj = self._hash_func(digest)

        if as_bytes:
            return hash_obj.digest()

        return int_from_hex(hash_obj.hexdigest())