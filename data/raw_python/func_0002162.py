def create_session(self):
        """Create a new HTTP session with our user-agent set.

        Returns
        -------
        session : requests.Session
            The created session

        See Also
        --------
        urlopen, set_session_options

        """
        ret = requests.Session()
        ret.headers['User-Agent'] = self.user_agent
        for k, v in self.options.items():
            setattr(ret, k, v)
        return ret