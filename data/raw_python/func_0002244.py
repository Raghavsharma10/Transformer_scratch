def login(self):
        """ Login to verisure app api

        Login before calling any read or write commands

        """
        if os.path.exists(self._cookieFileName):
            with open(self._cookieFileName, 'r') as cookieFile:
                self._vid = cookieFile.read().strip()

            try:
                self._get_installations()
            except ResponseError:
                self._vid = None
                os.remove(self._cookieFileName)

        if self._vid is None:
            self._create_cookie()
            with open(self._cookieFileName, 'w') as cookieFile:
                cookieFile.write(self._vid)
            self._get_installations()

        self._giid = self.installations[0]['giid']