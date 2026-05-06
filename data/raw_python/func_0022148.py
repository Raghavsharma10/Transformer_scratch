def set_debug(self, debuglevel):
        """
        Change the debug level of the API

        **Returns:** No item returned.
        """
        if isinstance(debuglevel, int):
            self._debuglevel = debuglevel

        if self._debuglevel == 1:
            logging.basicConfig(level=logging.INFO,
                                format="%(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s")
            api_logger.setLevel(logging.INFO)
        elif self._debuglevel == 2:
            logging.basicConfig(level=logging.DEBUG,
                                format="%(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s")
            requests.cookies.cookielib.debug = True
            api_logger.setLevel(logging.DEBUG)
        elif self._debuglevel >= 3:
            logging.basicConfig(level=logging.DEBUG,
                                format="%(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s")
            requests.cookies.cookielib.debug = True
            api_logger.setLevel(logging.DEBUG)
            urllib3_logger = logging.getLogger("requests.packages.urllib3")
            urllib3_logger.setLevel(logging.DEBUG)
            urllib3_logger.propagate = True
        else:
            # Remove all handlers
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            # set logging level to default
            requests.cookies.cookielib.debug = False
            api_logger.setLevel(logging.WARNING)

        return