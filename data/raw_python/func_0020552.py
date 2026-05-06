def __kill_if_not_shut_properly(self):
        """
        If the HoverFly process on these given ports
        did not shut down correctly, then kill the pid
        before launching a new instance.
        todo: this will kill existing HoverFly processes
        on those ports indiscriminately
        """
        import tempfile
        d = tempfile.gettempdir()
        name = os.path.join(d, "hoverpy.%i.%i"%(self._proxyPort, self._adminPort))
        if os.path.exists(name):
            logging.debug("pid file exists.. killing it")
            f = open(name, "r")
            pid = int(f.read())
            try:
                import signal
                os.kill(pid, signal.SIGTERM)
                logging.debug("killing %i"%pid)
            except:
                logging.debug("nothing to clean up")
                pass
            finally:
                os.unlink(name)