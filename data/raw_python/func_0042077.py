def watch(static_root, watch_paths=None, on_reload=None, host='localhost', port=5555, server_base_path="/",
          watcher_interval=1.0, recursive=True, open_browser=True, open_browser_delay=1.0):
    """Initialises an HttpWatcherServer to watch the given path for changes. Watches until the IO loop
    is terminated, or a keyboard interrupt is intercepted.

    Args:
        static_root: The path whose contents are to be served and watched.
        watch_paths: The paths to be watched for changes. If not supplied, this defaults to the static root.
        on_reload: An optional callback to pass to the watcher server that will be executed just before the
            server triggers a reload in connected clients.
        host: The host to which to bind our server.
        port: The port to which to bind our server.
        server_base_path: If the content is to be served from a non-standard base path, specify it here.
        watcher_interval: The maximum refresh rate of the watcher server.
        recursive: Whether to monitor the watch path recursively.
        open_browser: Whether or not to automatically attempt to open the user's browser at the root URL of
            the project (default: True).
        open_browser_delay: The number of seconds to wait before attempting to open the user's browser.
    """
    server = httpwatcher.HttpWatcherServer(
        static_root,
        watch_paths=watch_paths,
        on_reload=on_reload,
        host=host,
        port=port,
        server_base_path=server_base_path,
        watcher_interval=watcher_interval,
        recursive=recursive,
        open_browser=open_browser,
        open_browser_delay=open_browser_delay
    )
    server.listen()

    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        server.shutdown()