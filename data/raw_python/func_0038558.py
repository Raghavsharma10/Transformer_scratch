def init_ixn(api, logger, install_dir=None):
    """ Create IXN object.

    :param api: tcl/python/rest
    :type api: trafficgenerator.tgn_utils.ApiType
    :param logger: logger object
    :param install_dir: IXN installation directory
    :return: IXN object
    """

    if api == ApiType.tcl:
        api_wrapper = IxnTclWrapper(logger, install_dir)
    elif api == ApiType.python:
        api_wrapper = IxnPythonWrapper(logger, install_dir)
    elif api == ApiType.rest:
        api_wrapper = IxnRestWrapper(logger)
    else:
        raise TgnError('{} API not supported - use Tcl, python or REST'.format(api))
    return IxnApp(logger, api_wrapper)