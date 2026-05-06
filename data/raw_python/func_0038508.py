def cli(**settings):
    """Notify about new reviews in AppStore and Google Play in slack.

       Launch command using supervisor or using screen/tmux/etc.
       Reviews are fetched for multiple apps and languages in --beat=300 interval.
    """
    setup_logging(settings)
    settings = setup_languages(settings)
    channels = setup_channel_map(settings)
    app = CriticApp(**dict(settings, channels=channels))
    if settings['sentry_dsn']:
        app.sentry_client = Client(settings['sentry_dsn'])
        logger.debug('Errors are reported to %s' % settings['sentry_dsn'])
    else:
        app.sentry_client = None

    if settings['version']:
        click.echo('Version %s' % critics.__version__)
        return
    if not (settings['ios'] or settings['android']):
        click.echo('Please choose either --ios or --android')
        return

    loop = tornado.ioloop.IOLoop.instance()

    if app.load_model():
        logger.debug('Model loaded OK, not skipping notify on first run')
        notify = True
    else:
        notify = False

    if settings['ios']:
        logger.info('Tracking IOS apps: %s', ', '.join(settings['ios']))
        itunes = tornado.ioloop.PeriodicCallback(partial(app.poll_store, 'ios'),
                                                 1000 * settings['beat'], loop)
        itunes.start()
    if settings['android']:
        logger.info('Tracking Android apps: %s', ', '.join(settings['android']))
        google_play = tornado.ioloop.PeriodicCallback(partial(app.poll_store, 'android'),
                                                      1000 * settings['beat'], loop)
        google_play.start()

    echo_channel_map(channels)

    if settings['ios']:
        app.poll_store('ios', notify=notify)
    if settings['android']:
        app.poll_store('android', notify=notify)

    if settings['stats']:
        port = int(settings['stats'])
        logger.debug('Serving metrics server on port %s' % port)
        start_http_server(port)

    if settings['daemonize']:
        loop.start()