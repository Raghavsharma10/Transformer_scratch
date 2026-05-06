def asAccessibleTo(self, query):
        """
        @param query: An Axiom query describing the Items to retrieve, which this
        role can access.
        @type query: an L{iaxiom.IQuery} provider.

        @return: an iterable which yields the shared proxies that are available
        to the given role, from the given query.
        """
        # XXX TODO #2371: this method really *should* be returning an L{IQuery}
        # provider as well, but that is kind of tricky to do.  Currently, doing
        # queries leaks authority, because the resulting objects have stores
        # and "real" items as part of their interface; having this be a "real"
        # query provider would obviate the need to escape the L{SharedProxy}
        # security constraints in order to do any querying.
        allRoles = list(self.allRoles())
        count = 0
        unlimited = query.cloneQuery(limit=None)
        for result in unlimited:
            allShares = list(query.store.query(
                    Share,
                    AND(Share.sharedItem == result,
                        Share.sharedTo.oneOf(allRoles))))
            interfaces = []
            for share in allShares:
                interfaces += share.sharedInterfaces
            if allShares:
                count += 1
                yield SharedProxy(result, interfaces, allShares[0].shareID)
                if count == query.limit:
                    return