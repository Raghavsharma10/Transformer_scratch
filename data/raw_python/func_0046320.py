def env(self, **kw):
        '''
        Allows adding/overriding env vars in the execution context.
        :param kw: Key-value pairs
        :return: self
        '''
        self._original_env = kw
        if self._env is None:
            self._env = dict(os.environ)
        self._env.update({k: unicode(v) for k, v in kw.iteritems()})
        return self