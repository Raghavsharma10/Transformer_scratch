def select_logfile(self, logfile):
        """
        Parameters
        ----------
        logfile : str

        Returns
        -------
        dict
        """
        data = 'logFileSelect,' + logfile
        r = self._basic_post(url='logBrowser', data=data)
        return r.json()