def sadd(self, name, values, *args):
        """
        Add the specified members to the Set.

        :param name: str     the name of the redis key
        :param values: a list of values or a simple value.
        :return: Future()
        """
        with self.pipe as pipe:
            values = [self.valueparse.encode(v) for v in
                      self._parse_values(values, args)]
            return pipe.sadd(self.redis_key(name), *values)