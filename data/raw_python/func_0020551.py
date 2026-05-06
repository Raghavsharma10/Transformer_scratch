def __rmpid(self):
        """
        Remove the PID file on shutdown, unfortunately
        this may not get called if not given the time to
        shut down.
        """
        import tempfile
        d = tempfile.gettempdir()
        name = os.path.join(d, "hoverpy.%i.%i"%(self._proxyPort, self._adminPort))
        if os.path.exists(name):
            os.unlink(name)
            logging.debug("deleting %s"%name)