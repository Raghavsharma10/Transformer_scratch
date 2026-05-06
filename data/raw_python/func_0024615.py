def setup_apiv2():
    """
    Setup apiv2 when using PyQt4 and Python2.
    """
    # setup PyQt api to version 2
    if sys.version_info[0] == 2:
        logging.getLogger(__name__).debug(
            'setting up SIP API to version 2')
        import sip
        try:
            sip.setapi("QString", 2)
            sip.setapi("QVariant", 2)
        except ValueError:
            logging.getLogger(__name__).critical(
                "failed to set up sip api to version 2 for PyQt4")
            raise ImportError('PyQt4')