def indirect(self, interface):
        """
        Create an L{IConchUser} avatar which will use L{ShellServer} to
        interact with the connection.
        """
        if interface is IConchUser:
            componentized = Componentized()

            user = _BetterTerminalUser(componentized, None)
            session = _BetterTerminalSession(componentized)
            session.transportFactory = TerminalSessionTransport
            session.chainedProtocolFactory = lambda: ServerProtocol(ShellServer, self.store)

            componentized.setComponent(IConchUser, user)
            componentized.setComponent(ISession, session)

            return user

        raise NotImplementedError(interface)