def app(session_id, app, bind, port):
    """
    Run a local proxy to a service provided by Backend.AI compute sessions.

    The type of proxy depends on the app definition: plain TCP or HTTP.

    \b
    SESSID: The compute session ID.
    APP: The name of service provided by the given session.
    """
    api_session = None
    runner = None

    async def app_setup():
        nonlocal api_session, runner
        loop = current_loop()
        api_session = AsyncSession()
        # TODO: generalize protocol using service ports metadata
        protocol = 'http'
        runner = ProxyRunner(api_session, session_id, app,
                             protocol, bind, port,
                             loop=loop)
        await runner.ready()
        print_info(
            "A local proxy to the application \"{0}\" ".format(app) +
            "provided by the session \"{0}\" ".format(session_id) +
            "is available at: {0}://{1}:{2}"
            .format(protocol, bind, port)
        )

    async def app_shutdown():
        nonlocal api_session, runner
        print_info("Shutting down....")
        await runner.close()
        await api_session.close()
        print_info("The local proxy to \"{}\" has terminated."
                   .format(app))

    asyncio_run_forever(app_setup(), app_shutdown(),
                        stop_signals={signal.SIGINT, signal.SIGTERM})