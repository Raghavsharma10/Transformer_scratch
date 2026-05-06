def fromSharedItem(cls, sharedItem):
        """
        Return an instance of C{cls} derived from the given L{Item} that has
        been shared.

        Note that this API does not provide any guarantees of which result it
        will choose.  If there are are multiple possible return values, it will
        select and return only one.  Items may be shared under multiple
        L{shareID}s.  A user may have multiple valid account names.  It is
        sometimes impossible to tell from context which one is appropriate, so
        if your application has another way to select a specific shareID you
        should use that instead.

        @param sharedItem: an L{Item} that should be shared.

        @return: an L{Identifier} describing the C{sharedItem} parameter.

        @raise L{NoSuchShare}: if the given item is not shared or its store
        does not contain any L{LoginMethod} items which would identify a user.
        """
        localpart = None
        for (localpart, domain) in userbase.getAccountNames(sharedItem.store):
            break
        if localpart is None:
            raise NoSuchShare()
        for share in sharedItem.store.query(Share,
                                            Share.sharedItem == sharedItem):
            break
        else:
            raise NoSuchShare()
        return cls(
            shareID=share.shareID,
            localpart=localpart, domain=domain)