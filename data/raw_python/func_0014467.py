def connect(self, hostkey=None, username='', password=None, pkey=None):
        """
        Negotiate an SSH2 session, and optionally verify the server's host key
        and authenticate using a password or private key.  This is a shortcut
        for L{start_client}, L{get_remote_server_key}, and
        L{Transport.auth_password} or L{Transport.auth_publickey}.  Use those
        methods if you want more control.

        You can use this method immediately after creating a Transport to
        negotiate encryption with a server.  If it fails, an exception will be
        thrown.  On success, the method will return cleanly, and an encrypted
        session exists.  You may immediately call L{open_channel} or
        L{open_session} to get a L{Channel} object, which is used for data
        transfer.

        @note: If you fail to supply a password or private key, this method may
        succeed, but a subsequent L{open_channel} or L{open_session} call may
        fail because you haven't authenticated yet.

        @param hostkey: the host key expected from the server, or C{None} if
            you don't want to do host key verification.
        @type hostkey: L{PKey<pkey.PKey>}
        @param username: the username to authenticate as.
        @type username: str
        @param password: a password to use for authentication, if you want to
            use password authentication; otherwise C{None}.
        @type password: str
        @param pkey: a private key to use for authentication, if you want to
            use private key authentication; otherwise C{None}.
        @type pkey: L{PKey<pkey.PKey>}

        @raise SSHException: if the SSH2 negotiation fails, the host key
            supplied by the server is incorrect, or authentication fails.
        """
        if hostkey is not None:
            self._preferred_keys = [ hostkey.get_name() ]

        self.start_client()

        # check host key if we were given one
        if (hostkey is not None):
            key = self.get_remote_server_key()
            if (key.get_name() != hostkey.get_name()) or (str(key) != str(hostkey)):
                self._log(DEBUG, 'Bad host key from server')
                self._log(DEBUG, 'Expected: %s: %s' % (hostkey.get_name(), repr(str(hostkey))))
                self._log(DEBUG, 'Got     : %s: %s' % (key.get_name(), repr(str(key))))
                raise SSHException('Bad host key from server')
            self._log(DEBUG, 'Host key verified (%s)' % hostkey.get_name())

        if (pkey is not None) or (password is not None):
            if password is not None:
                self._log(DEBUG, 'Attempting password auth...')
                self.auth_password(username, password)
            else:
                self._log(DEBUG, 'Attempting public-key auth...')
                self.auth_publickey(username, pkey)

        return