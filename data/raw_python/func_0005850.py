def register_handler(self, target=None):
        """Decorator for a function to be used as a signal handler.

        :param str|unicode target: Where this signal will be delivered to. Default: ``worker``.

            * ``workers``  - run the signal handler on all the workers
            * ``workerN`` - run the signal handler only on worker N
            * ``worker``/``worker0`` - run the signal handler on the first available worker
            * ``active-workers`` - run the signal handlers on all the active [non-cheaped] workers

            * ``mules`` - run the signal handler on all of the mules
            * ``muleN`` - run the signal handler on mule N
            * ``mule``/``mule0`` - run the signal handler on the first available mule

            * ``spooler`` - run the signal on the first available spooler
            * ``farmN/farm_XXX``  - run the signal handler in the mule farm N or named XXX

            * http://uwsgi.readthedocs.io/en/latest/Signals.html#signals-targets

        """
        target = target or 'worker'
        sign_num = self.num

        def wrapper(func):

            _LOG.debug("Registering '%s' as signal '%s' handler ...", func.__name__, sign_num)

            uwsgi.register_signal(sign_num, target, func)

            return func

        return wrapper