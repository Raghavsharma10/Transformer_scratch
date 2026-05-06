def rpush(self, name, *values):
        """
        Push the value into the list from the *right* side

        :param name: str     the name of the redis key
        :param values: a list of values or single value to push
        :return: Future()
        """
        with self.pipe as pipe:
            v_encode = self.valueparse.encode
            values = [v_encode(v) for v in self._parse_values(values)]
            return pipe.rpush(self.redis_key(name), *values)