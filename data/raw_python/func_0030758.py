def main():
    """
    Start the AMP server and the reactor.
    """
    startLogging(stdout)
    checker = InMemoryUsernamePasswordDatabaseDontUse()
    checker.addUser("testuser", "examplepass")
    realm = AdditionRealm()
    factory = CredAMPServerFactory(Portal(realm, [checker]))
    reactor.listenTCP(7805, factory)
    reactor.run()