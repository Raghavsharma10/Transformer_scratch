def autodetect():
    """
    Auto-detects and use the first available QT_API by importing them in the
    following order:

    1) PyQt5
    2) PyQt4
    3) PySide
    """
    logging.getLogger(__name__).debug('auto-detecting QT_API')
    try:
        logging.getLogger(__name__).debug('trying PyQt5')
        import PyQt5
        os.environ[QT_API] = PYQT5_API[0]
        logging.getLogger(__name__).debug('imported PyQt5')
    except ImportError:
        try:
            logging.getLogger(__name__).debug('trying PyQt4')
            setup_apiv2()
            import PyQt4
            os.environ[QT_API] = PYQT4_API[0]
            logging.getLogger(__name__).debug('imported PyQt4')
        except ImportError:
            try:
                logging.getLogger(__name__).debug('trying PySide')
                import PySide
                os.environ[QT_API] = PYSIDE_API[0]
                logging.getLogger(__name__).debug('imported PySide')
            except ImportError:
                raise PythonQtError('No Qt bindings could be found')