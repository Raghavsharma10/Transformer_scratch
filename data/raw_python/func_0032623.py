def _getUsername(self):
        """
        Return a localpart@domain style string naming the owner of our store.

        @rtype: C{unicode}
        """
        for (l, d) in getAccountNames(self.store):
            return l + u'@' + d