def createPort(self, portNumber, ssl, certPath, factory, interface=u''):
        """
        Create a new listening port.

        @type portNumber: C{int}
        @param portNumber: Port number on which to listen.

        @type ssl: C{bool}
        @param ssl: Indicates whether this should be an SSL port or not.

        @type certPath: C{str}
        @param ssl: If C{ssl} is true, a path to a certificate file somewhere
        within the site store's files directory.  Ignored otherwise.

        @param factory: L{Item} which provides L{IProtocolFactoryFactory} which
        will be used to get a protocol factory to associate with this port.

        @return: C{None}
        """
        store = self.store.parent
        if ssl:
            port = SSLPort(store=store, portNumber=portNumber,
                           certificatePath=FilePath(certPath), factory=factory,
                           interface=interface)
        else:
            port = TCPPort(store=store, portNumber=portNumber, factory=factory,
                           interface=interface)
        installOn(port, store)