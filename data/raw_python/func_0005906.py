def env(self, key, value=None, unset=False, asap=False):
        """Processes (sets/unsets) environment variable.

        If is not given in `set` mode value will be taken from current env.

        :param str|unicode key:

        :param value:

        :param bool unset: Whether to unset this variable.

        :param bool asap: If True env variable will be set as soon as possible.

        """
        if unset:
            self._set('unenv', key, multi=True)
        else:
            if value is None:
                value = os.environ.get(key)

            self._set('%senv' % ('i' if asap else ''), '%s=%s' % (key, value), multi=True)

        return self