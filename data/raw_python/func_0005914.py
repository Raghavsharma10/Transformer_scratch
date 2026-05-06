def set_wsgi_params(self, module=None, callable_name=None, env_strategy=None):
        """Set wsgi related parameters.

        :param str|unicode module:
            * load .wsgi file as the Python application
            * load a WSGI module as the application.

            .. note:: The module (sans ``.py``) must be importable, ie. be in ``PYTHONPATH``.

            Examples:
                * mypackage.my_wsgi_module -- read from `application` attr of mypackage/my_wsgi_module.py
                * mypackage.my_wsgi_module:my_app -- read from `my_app` attr of mypackage/my_wsgi_module.py

        :param str|unicode callable_name: Set WSGI callable name. Default: application.

        :param str|unicode env_strategy: Strategy for allocating/deallocating
            the WSGI env, can be:

            * ``cheat`` - preallocates the env dictionary on uWSGI startup and clears it
                after each request. Default behaviour for uWSGI <= 2.0.x

            * ``holy`` - creates and destroys the environ dictionary at each request.
                Default behaviour for uWSGI >= 2.1

        """
        module = module or ''

        if '/' in module:
            self._set('wsgi-file', module, condition=module)

        else:
            self._set('wsgi', module, condition=module)

        self._set('callable', callable_name)
        self._set('wsgi-env-behaviour', env_strategy)

        return self._section