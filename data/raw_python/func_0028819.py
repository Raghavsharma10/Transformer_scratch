def boot(app_name) -> Rinzler:
    """
    Start Rinzler App
    :param app_name: str Application's identifier
    :return: dict
    """
    app = Rinzler(app_name)
    app.log.info("App booted =)")

    return app