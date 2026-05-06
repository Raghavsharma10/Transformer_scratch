def bind_to_env(self, bound_env):
        '''
        Get a copy of the reverse, bound to `env` object.
        Can be found in env.root attribute::

            # done in iktomi.web.app.Application
            env.root = Reverse.from_handler(app).bind_to_env(env)
        '''
        return self.__class__(self._scope, self._location,
                              path=self._path, host=self._host,
                              fragment=self._fragment,
                              ready=self._ready,
                              need_arguments=self._need_arguments,
                              finalize_params=self._finalize_params,
                              parent=self._parent,
                              bound_env=bound_env)