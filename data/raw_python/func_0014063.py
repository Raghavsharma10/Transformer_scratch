def setPendingKeyExchange(self, sequence, ourBaseKey, ourRatchetKey, ourIdentityKey):
        """
        :type sequence: int
        :type ourBaseKey: ECKeyPair
        :type ourRatchetKey: ECKeyPair
        :type  ourIdentityKey: IdentityKeyPair
        """
        structure = self.sessionStructure.PendingKeyExchange()
        structure.sequence = sequence
        structure.localBaseKey = ourBaseKey.getPublicKey().serialize()
        structure.localBaseKeyPrivate = ourBaseKey.getPrivateKey().serialize()
        structure.localRatchetKey = ourRatchetKey.getPublicKey().serialize()
        structure.localRatchetKeyPrivate = ourRatchetKey.getPrivateKey().serialize()
        structure.localIdentityKey = ourIdentityKey.getPublicKey().serialize()
        structure.localIdentityKeyPrivate = ourIdentityKey.getPrivateKey().serialize()

        self.sessionStructure.pendingKeyExchange.MergeFrom(structure)