def getShare(self, shareID):
        """
        Retrieve a proxy object for a given shareID, previously shared with
        this role or one of its group roles via L{Role.shareItem}.

        @return: a L{SharedProxy}.  This is a wrapper around the shared item
        which only exposes those interfaces explicitly allowed for the given
        role.

        @raise: L{NoSuchShare} if there is no item shared to the given role for
        the given shareID.
        """
        shares = list(
            self.store.query(Share,
                             AND(Share.shareID == shareID,
                                 Share.sharedTo.oneOf(self.allRoles()))))
        interfaces = []
        for share in shares:
            interfaces += share.sharedInterfaces
        if shares:
            return SharedProxy(shares[0].sharedItem,
                               interfaces,
                               shareID)
        raise NoSuchShare()