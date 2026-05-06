def get_wms_version_negotiate(url, timeout=10):
    """
    OWSLib wrapper function to perform version negotiation against owslib.wms.WebMapService
    """

    try:
        LOGGER.debug('Trying a WMS 1.3.0 GetCapabilities request')
        return WebMapService(url, version='1.3.0', timeout=timeout)
    except Exception as err:
        LOGGER.warning('WMS 1.3.0 support not found: %s', err)
        LOGGER.debug('Trying a WMS 1.1.1 GetCapabilities request instead')
        return WebMapService(url, version='1.1.1', timeout=timeout)