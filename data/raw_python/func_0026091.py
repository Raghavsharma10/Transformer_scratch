def wireHandlers(cfg):
    """
    If the device is configured to run against a remote server, ping that device on a scheduled basis with our current
    state.
    :param cfg: the config object.
    :return:
    """
    logger = logging.getLogger('recorder')
    httpPoster = cfg.handlers.get('remote')
    csvLogger = cfg.handlers.get('local')
    activeHandler = None
    if httpPoster is None:
        if csvLogger is None:
            logger.warning("App is running with discard handler only, ALL DATA WILL BE DISCARDED!!!")
        else:
            logger.info("App is running in standalone mode, logging data to local filesystem")
            activeHandler = csvLogger
    else:
        logger.info("App is running against remote server, logging data to " + httpPoster.target)
        activeHandler = httpPoster
        heartbeater.serverURL = httpPoster.target
        heartbeater.ping()

    if activeHandler is not None:
        for device in cfg.recordingDevices.values():
            if activeHandler is httpPoster:
                httpPoster.deviceName = device.name
            copied = copy.copy(activeHandler)
            device.dataHandler = copied if not cfg.useAsyncHandlers else AsyncHandler('recorder', copied)