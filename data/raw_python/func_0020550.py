def __writepid(self, pid):
        """
        HoverFly fails to launch if it's already running on
        the same ports. So we have to keep track of them using
        temp files with the proxy port and admin port, containing
        the processe's PID. 
        """
        import tempfile
        d = tempfile.gettempdir()
        name = os.path.join(d, "hoverpy.%i.%i"%(self._proxyPort, self._adminPort))
        with open(name, 'w') as f:
            f.write(str(pid))
            logging.debug("writing to %s"%name)