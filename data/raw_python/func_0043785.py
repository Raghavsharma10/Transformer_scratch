def check(self, check_url=None):
        """
        Checks whether a server is running.

        :param str check_url:
            URL where to check whether the server is running.
            Default is ``"http://{self.host}:{self.port}"``.
        """

        if check_url is not None:
            self.check_url = self._normalize_check_url(check_url)

        response = None
        sleeped = 0.0
        t = datetime.now()

        while not response:
            try:
                response = requests.get(self.check_url, verify=False)
            except requests.exceptions.ConnectionError:
                if sleeped > self.timeout:
                    self._kill()
                    raise LiveAndLetDieError(
                        '{0} server {1} didn\'t start in specified timeout {2} '
                        'seconds!\ncommand: {3}'.format(
                            self.__class__.__name__,
                            self.check_url,
                            self.timeout,
                            ' '.join(self.create_command())
                        )
                    )
                time.sleep(1)
                sleeped = _get_total_seconds(datetime.now() - t)

        return _get_total_seconds(datetime.now() - t)