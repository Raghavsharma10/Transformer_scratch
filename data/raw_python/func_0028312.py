def proxy(ctx, bind, port):
    """
    Run a non-encrypted non-authorized API proxy server.
    Use this only for development and testing!
    """
    app = web.Application()
    app.on_startup.append(startup_proxy)
    app.on_cleanup.append(cleanup_proxy)

    app.router.add_route("GET", r'/stream/{path:.*$}', websocket_handler)
    app.router.add_route("GET", r'/wsproxy/{path:.*$}', websocket_handler)
    app.router.add_route('*', r'/{path:.*$}', web_handler)
    if getattr(ctx.args, 'testing', False):
        return app
    web.run_app(app, host=bind, port=port)