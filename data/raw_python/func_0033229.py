def server():
    """Runs the server"""

    tornado.log.enable_pretty_logging()

    # Get and validate the server_type
    server_type = oz.settings["server_type"]
    if server_type not in [None, "wsgi", "asyncio", "twisted"]:
        raise Exception("Unknown server type: %s" % server_type)

    # Install the correct ioloop if necessary
    if server_type == "asyncio":
        from tornado.platform.asyncio import AsyncIOMainLoop
        AsyncIOMainLoop().install()
    elif server_type == "twisted":
        from tornado.platform.twisted import TwistedIOLoop
        TwistedIOLoop().install()

    if server_type == "wsgi":
        wsgi_app = tornado.wsgi.WSGIApplication(oz._routes, **oz.settings)
        wsgi_srv = wsgiref.simple_server.make_server("", oz.settings["port"], wsgi_app)
        wsgi_srv.serve_forever()
    else:
        web_app = tornado.web.Application(oz._routes, **oz.settings)

        if oz.settings["ssl_cert_file"] != None and oz.settings["ssl_key_file"] != None:
            ssl_options = {
                "certfile": oz.settings["ssl_cert_file"],
                "keyfile": oz.settings["ssl_key_file"],
                "cert_reqs": oz.settings["ssl_cert_reqs"],
                "ca_certs": oz.settings["ssl_ca_certs"]
            }
        else:
            ssl_options = None

        http_srv = tornado.httpserver.HTTPServer(
            web_app,
            ssl_options=ssl_options,
            body_timeout=oz.settings["body_timeout"],
            xheaders=oz.settings["xheaders"]
        )

        http_srv.bind(oz.settings["port"])

        server_workers = oz.settings["server_workers"]

        if server_workers > 1:
            if oz.settings["debug"]:
                print("WARNING: Debug is enabled, but multiple server workers have been configured. Only one server worker can run in debug mode.")
                server_workers = 1
            elif (server_type == "asyncio" or server_type == "twisted"):
                print("WARNING: A non-default server type is being used, but multiple server workers have been configured. Only one server worker can run on a non-default server type.")
                server_workers = 1

        # Forks multiple sub-processes if server_workers > 1
        http_srv.start(server_workers)

        # Registers signal handles for graceful server shutdown
        if oz.settings.get("use_graceful_shutdown"):
            if server_type == "asyncio" or server_type == "twisted":
                print("WARNING: Cannot enable graceful shutdown for asyncio or twisted server types.")
            else:
                # NOTE: Do not expect any logging to with certain tools (e.g., invoker),
                # because they may quiet logs on SIGINT/SIGTERM
                signal.signal(signal.SIGTERM, functools.partial(_shutdown_tornado_ioloop, http_srv))
                signal.signal(signal.SIGINT, functools.partial(_shutdown_tornado_ioloop, http_srv))

        # Starts the ioloops
        if server_type == "asyncio":
            import asyncio
            asyncio.get_event_loop().run_forever()
        elif server_type == "twisted":
            from twisted.internet import reactor
            reactor.run()
        else:
            from tornado import ioloop
            ioloop.IOLoop.instance().start()