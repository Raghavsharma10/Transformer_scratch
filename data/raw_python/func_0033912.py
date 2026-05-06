def gcommer_donate_threaded(interval=5, region='EU-London', mode=None):
    """
    Run a daemon thread that requests and
    donates a token every `interval` seconds.
    """
    def donate_thread():
        while 1:
            gcommer_donate(*find_server(region, mode))
            time.sleep(interval)

    Thread(target=donate_thread, daemon=True).start()