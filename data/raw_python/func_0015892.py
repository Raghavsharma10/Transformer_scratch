def get_whoami(self):
        """
        A convenience function used in the event that you need to confirm that
        the broker thinks you are who you think you are.

        :returns dict whoami: Dict structure contains:
            * administrator: whether the user is has admin privileges
            * name: user name
            * auth_backend: backend used to determine admin rights
        """
        path = Client.urls['whoami']
        whoami = self._call(path, 'GET')
        return whoami