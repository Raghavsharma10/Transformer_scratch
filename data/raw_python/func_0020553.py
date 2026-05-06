def __start(self):
        """
        Start the hoverfly process.

        This function waits until it can make contact
        with the hoverfly API before returning.
        """
        logging.debug("starting %i" % id(self))
        self.__kill_if_not_shut_properly()
        self.FNULL = open(os.devnull, 'w')
        flags = self.__flags()
        cmd = [hoverfly] + flags
        if self._showCmd:
            print(cmd)
        self._process = Popen(
            [hoverfly] +
            flags,
            stdin=self.FNULL,
            stdout=self.FNULL,
            stderr=subprocess.STDOUT)
        start = time.time()
        while time.time() - start < 1:
            try:
                url = "http://%s:%i/api/health" % (self._host, self._adminPort)
                r = self._session.get(url)
                j = r.json()
                up = "message" in j and "healthy" in j["message"]
                if up:
                    logging.debug("has pid %i" % self._process.pid)
                    self.__writepid(self._process.pid)
                    return self._process
                else:
                    time.sleep(1/100.0)
            except:
                # import traceback
                # traceback.print_exc()
                # wait 10 ms before trying again
                time.sleep(1/100.0)
                pass

        logging.error("Could not start hoverfly!")
        raise ValueError("Could not start hoverfly!")