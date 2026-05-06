def load_iterable(self, iterable, session=None):
        '''Load an ``iterable``.

        By default it returns a generator of data loaded via the
        :meth:`loads` method.

        :param iterable: an iterable over data to load.
        :param session: Optional :class:`stdnet.odm.Session`.
        :return: an iterable over decoded data.
        '''
        data = []
        load = self.loads
        for v in iterable:
            data.append(load(v))
        return data