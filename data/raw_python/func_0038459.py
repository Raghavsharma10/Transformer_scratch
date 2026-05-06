def set_server(self, wsgi_app, fnc_serve=None):
        """
        figures out how the wsgi application is to be served
        according to config
        """

        self.set_wsgi_app(wsgi_app)

        ssl_config = self.get_config("ssl")
        ssl_context = {}

        if self.get_config("server") == "gevent":

            if ssl_config.get("enabled"):
                ssl_context["certfile"] = ssl_config.get("cert")
                ssl_context["keyfile"] = ssl_config.get("key")

            from gevent.pywsgi import WSGIServer

            http_server = WSGIServer(
                (self.host, self.port),
                wsgi_app,
                **ssl_context
            )

            self.log.debug("Serving WSGI via gevent.pywsgi.WSGIServer")

            fnc_serve = http_server.serve_forever

        elif self.get_config("server") == "uwsgi":
            self.pluginmgr_config["start_manual"] = True

        elif self.get_config("server") == "gunicorn":
            self.pluginmgr_config["start_manual"] = True

        elif self.get_config("server") == "self":
            fnc_serve = self.run

        # figure out async handler

        if self.get_config("async") == "gevent":

            # handle async via gevent
            import gevent

            self.log.debug("Handling wsgi on gevent")

            self.worker = gevent.spawn(fnc_serve)

        elif self.get_config("async") == "thread":

            self.worker = fnc_serve

        else:

            self.worker = fnc_serve