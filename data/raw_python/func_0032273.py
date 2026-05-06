def _add_extra_handlers(self, handlers):
        """
        Adds the extra handler (defined by the user)

        :param handlers: a list of :py:class:`tornado.web.RequestHandler` instances.
        :return:
        """
        extra_handlers = [(h[0], h[1], {"microservice": self}) for h in self.extra_handlers]
        handlers.extend(extra_handlers)