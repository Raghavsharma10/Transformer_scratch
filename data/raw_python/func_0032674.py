def eval(self, script, numkeys, *keys_and_args):
        """
        Run a lua script against the key.
        Doesn't support multi-key lua operations because
        we wouldn't be able to know what argument to namespace.
        Also, redis cluster doesn't really support multi-key operations.

        :param script: str  A lua script targeting the current key.
        :param numkeys: number of keys passed to the script
        :param keys_and_args: list of keys and args passed to script
        :return: Future()
        """
        with self.pipe as pipe:
            keys_and_args = [a if i >= numkeys else self.redis_key(a) for i, a
                             in enumerate(keys_and_args)]
            return pipe.eval(script, numkeys, *keys_and_args)