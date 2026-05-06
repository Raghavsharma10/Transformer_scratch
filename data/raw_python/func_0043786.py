def live(self, kill_port=False, check_url=None):
        """
        Starts a live server in a separate process
        and checks whether it is running.

        :param bool kill_port:
            If ``True``, processes running on the same port as ``self.port``
            will be killed.

        :param str check_url:
            URL where to check whether the server is running.
            Default is ``"http://{self.host}:{self.port}"``.
        """
        
        pid = port_in_use(self.port, kill_port)

        if pid:
            raise LiveAndLetDieError(
                'Port {0} is already being used by process {1}!'
                .format(self.port, pid)
            )

        host = str(self.host)
        if re.match(_VALID_HOST_PATTERN, host):
            with open(os.devnull, "w") as devnull:
                if self.suppress_output:
                    self.process = subprocess.Popen(self.create_command(),
                                                    stderr=devnull,
                                                    stdout=devnull,
                                                    preexec_fn=os.setsid)
                else:
                    self.process = subprocess.Popen(self.create_command(),
                                                    preexec_fn=os.setsid)

            _log(self.logging, 'Starting process PID: {0}'
                 .format(self.process.pid))
            duration = self.check(check_url)
            _log(self.logging,
                 'Live server started in {0} seconds. PID: {1}'
                 .format(duration, self.process.pid))
            return self.process
        else:
            raise LiveAndLetDieError('{0} is not a valid host!'.format(host))