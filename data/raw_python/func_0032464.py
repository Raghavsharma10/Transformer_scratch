def getFactory(self):
        """
        Create an L{SSHFactory} which allows access to Mantissa accounts.
        """
        privateKey = Key.fromString(data=self.hostKey)
        public = privateKey.public()
        factory = SSHFactory()
        factory.publicKeys = {'ssh-rsa': public}
        factory.privateKeys = {'ssh-rsa': privateKey}
        factory.portal = Portal(
            IRealm(self.store), [ICredentialsChecker(self.store)])
        return factory