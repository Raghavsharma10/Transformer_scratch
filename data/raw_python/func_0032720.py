def zadd(self, name, members, score=1, nx=False,
             xx=False, ch=False, incr=False):
        """
        Add members in the set and assign them the score.

        :param name: str     the name of the redis key
        :param members: a list of item or a single item
        :param score: the score the assign to the item(s)
        :param nx:
        :param xx:
        :param ch:
        :param incr:
        :return: Future()
        """

        if nx:
            _args = ['NX']
        elif xx:
            _args = ['XX']
        else:
            _args = []

        if ch:
            _args.append('CH')

        if incr:
            _args.append('INCR')

        if isinstance(members, dict):
            for member, score in members.items():
                _args += [score, self.valueparse.encode(member)]
        else:
            _args += [score, self.valueparse.encode(members)]

        if nx and xx:
            raise InvalidOperation('cannot specify nx and xx at the same time')
        with self.pipe as pipe:
            return pipe.execute_command('ZADD', self.redis_key(name), *_args)