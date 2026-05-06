def set(self, name, value, ex=None, px=None, nx=False, xx=False):
        """
        Set the value at key ``name`` to ``value``

        ``ex`` sets an expire flag on key ``name`` for ``ex`` seconds.

        ``px`` sets an expire flag on key ``name`` for ``px`` milliseconds.

        ``nx`` if set to True, set the value at key ``name`` to ``value`` if it
        does not already exist.

        ``xx`` if set to True, set the value at key ``name`` to ``value`` if it
        already exists.

        :return: Future()
        """
        with self.pipe as pipe:
            value = self.valueparse.encode(value)
            return pipe.set(self.redis_key(name), value,
                            ex=ex, px=px, nx=nx, xx=xx)