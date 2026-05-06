def _setup_authentication(self, username, password):
        '''Create the authentication object with the given credentials.'''

        ## BUG WORKAROUND
        if self.version < 1.1:
            # Version 1.0 had a bug when using the key parameter.
            # Later versions have the opposite bug (a key in the username doesn't function)
            if not username:
                username = self._key
                self._key = None

        if not username:
            return
        if not password:
            password = '12345'  #the same combination on my luggage!  (required dummy value)

        #realm = 'Redmine API' - doesn't always work
        # create a password manager
        password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()

        password_mgr.add_password(None, self._url, username, password )
        handler = urllib2.HTTPBasicAuthHandler( password_mgr )

        # create "opener" (OpenerDirector instance)
        self._opener = urllib2.build_opener( handler )

        # set the opener when we fetch the URL
        self._opener.open( self._url )

        # Install the opener.
        urllib2.install_opener( self._opener )