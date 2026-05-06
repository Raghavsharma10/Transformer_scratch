def locateChild(self, ctx, segments):
        """
        Look up a shared item for the role viewing this SharingIndex and return a
        L{PublicAthenaLivePage} containing that shared item's fragment to the
        user.

        These semantics are UNSTABLE.  This method is adequate for simple uses,
        but it should be expanded in the future to be more consistent with
        other resource lookups.  In particular, it should allow share
        implementors to adapt their shares to L{IResource} directly rather than
        L{INavigableFragment}, to allow for simpler child dispatch.

        @param segments: a list of strings, the first of which should be the
        shareID of the desired item.

        @param ctx: unused.

        @return: a L{PublicAthenaLivePage} wrapping a customized fragment.
        """
        shareID = segments[0].decode('utf-8')

        role = self.webViewer.roleIn(self.userStore)

        # if there is an empty segment
        if shareID == u'':
            # then we want to return the default share.  if we find one, then
            # let's use that
            defaultShareID = getDefaultShareID(self.userStore)
            try:
                sharedItem = role.getShare(defaultShareID)
            except sharing.NoSuchShare:
                return rend.NotFound
        # otherwise the user is trying to access some other share
        else:
            # let's see if it's a real share
            try:
                sharedItem = role.getShare(shareID)
            # oops it's not
            except sharing.NoSuchShare:
                return rend.NotFound

        return (self.webViewer.wrapModel(sharedItem),
                segments[1:])