def shareItem(self, sharedItem, shareID=None, interfaces=ALL_IMPLEMENTED):
        """
        Share an item with this role.  This provides a way to expose items to
        users for later retrieval with L{Role.getShare}.

        @param sharedItem: an item to be shared.

        @param shareID: a unicode string.  If provided, specify the ID under which
        the shared item will be shared.

        @param interfaces: a list of Interface objects which specify the methods
        and attributes accessible to C{toRole} on C{sharedItem}.

        @return: a L{Share} which records the ability of the given role to
        access the given item.
        """
        if shareID is None:
            shareID = genShareID(sharedItem.store)
        return Share(store=self.store,
                     shareID=shareID,
                     sharedItem=sharedItem,
                     sharedTo=self,
                     sharedInterfaces=interfaces)