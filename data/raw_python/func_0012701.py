def start_sikuli_process(self, port=None):
        """
        This keyword is used to start sikuli java process.
        If library is inited with mode "OLD", sikuli java process is started automatically.
        If library is inited with mode "NEW", this keyword should be used.

        :param port: port of sikuli java process, if value is None or 0, a random free port will be used
        :return: None
        """
        if port is None or int(port) == 0:
            port = self._get_free_tcp_port()
        self.port = port
        start_retries = 0
        started = False
        while start_retries < 5:
            try:
                self._start_sikuli_java_process()
            except RuntimeError as err:
                print('error........%s' % err)
                if self.process:
                    self.process.terminate_process()
                self.port = self._get_free_tcp_port()
                start_retries += 1
                continue
            started = True
            break
        if not started:
            raise RuntimeError('Start sikuli java process failed!')
        self.remote = self._connect_remote_library()