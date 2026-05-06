def from_transport(cls, t):
        """
        Create an SFTP client channel from an open L{Transport}.

        @param t: an open L{Transport} which is already authenticated
        @type t: L{Transport}
        @return: a new L{SFTPClient} object, referring to an sftp session
            (channel) across the transport
        @rtype: L{SFTPClient}
        """
        chan = t.open_session()
        if chan is None:
            return None
        chan.invoke_subsystem('sftp')
        return cls(chan)