def runserver(host=None, port=None):
    """
    Run Tornado server
    """
    host = host or os.getenv('HTTP_HOST', '0.0.0.0')
    port = port or os.getenv('HTTP_PORT', '9001')
    zioloop = ioloop.IOLoop.instance()

    # setup pika client:
    pc = QueueManager(zioloop)
    app.pc = pc
    pc.connect()
    app.listen(port, host)
    zioloop.start()