def execute(self):
        """
        execute the commands inside the nested pipeline.
        This causes all queued up commands to be passed upstream to the
        parent, including callbacks.
        The state of this pipeline object gets cleaned up.
        :return:
        """
        stack = self._stack
        callbacks = self._callbacks
        self._stack = []
        self._callbacks = []

        deferred = []

        build = self._nested_future

        pipe = self._pipeline(self.connection_name)
        for item, args, kwargs, ref in stack:
            f = getattr(pipe, item)
            deferred.append(build(f(*args, **kwargs), ref))

        inject_callbacks = getattr(self.parent, '_inject_callbacks')
        inject_callbacks(deferred + callbacks)