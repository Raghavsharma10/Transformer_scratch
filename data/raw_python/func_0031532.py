def init_ixe(logger, host, port=4555, rsa_id=None):
    """ Connect to Tcl Server and Create IxExplorer object.

    :param logger: python logger object
    :param host: host (IxTclServer) IP address
    :param port: Tcl Server port
    :param rsa_id: full path to RSA ID file for Linux based IxVM
    :return: IXE object
    """

    return IxeApp(logger, IxTclHalApi(TclClient(logger, host, port, rsa_id)))