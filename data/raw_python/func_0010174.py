def start_server():
    """Starts up the imolecule server, complete with argparse handling."""
    parser = argparse.ArgumentParser(description="Opens a browser-based "
                                     "client that interfaces with the "
                                     "chemical format converter.")
    parser.add_argument('--debug', action="store_true", help="Prints all "
                        "transmitted data streams.")
    parser.add_argument('--port', type=int, default=8000, help="The port "
                        "on which to serve the website.")
    parser.add_argument('--timeout', type=int, default=5, help="The maximum "
                        "time, in seconds, allowed for a process to run "
                        "before returning an error.")
    parser.add_argument('--workers', type=int, default=2, help="The number of "
                        "worker processes to use with the server.")
    parser.add_argument('--no-browser', action="store_true", help="Disables "
                        "opening a browser window on startup.")
    global args
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    handlers = [(r'/', IndexHandler), (r'/websocket', WebSocket),
                (r'/static/(.*)', tornado.web.StaticFileHandler,
                 {'path': os.path.normpath(os.path.dirname(__file__))})]
    application = tornado.web.Application(handlers)
    application.listen(args.port)

    if not args.no_browser:
        webbrowser.open('http://localhost:%d/' % args.port, new=2)

    try:
        tornado.ioloop.IOLoop.instance().start()
    except KeyboardInterrupt:
        sys.stderr.write("Received keyboard interrupt. Stopping server.\n")
        tornado.ioloop.IOLoop.instance().stop()
        sys.exit(1)