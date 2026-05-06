def make_tornado_app(self):
        """
        Creates a :py:class`tornado.web.Application` instance that respect the
        JSON RPC 2.0 specs and exposes the designated methods. Can be used
        in tests to obtain the Tornado application.

        :return: a :py:class:`tornado.web.Application` instance
        """

        handlers = [
            (self.endpoint, TornadoJsonRpcHandler, {"microservice": self})
        ]

        self._add_extra_handlers(handlers)
        self._add_static_handlers(handlers)

        return Application(handlers, template_path=self.template_dir)