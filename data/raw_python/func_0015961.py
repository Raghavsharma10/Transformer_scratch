def logon(self, password='admin'):
        """
        Parameters
        ----------
        password : str
            default 'admin'

        Returns
        -------
        dict
        """
        r = self._basic_post(url='logon', data=password)
        return r.json()