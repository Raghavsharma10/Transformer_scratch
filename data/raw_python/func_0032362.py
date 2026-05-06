def _deleteTrackers(self, trackers):
        """
        Delete the given signup trackers and their associated signup resources.

        @param trackers: sequence of L{_SignupTrackers}
        """

        for tracker in trackers:
            if tracker.store is None:
                # we're not updating the list of live signups client side, so
                # we might get a signup that has already been deleted
                continue

            sig = tracker.signupItem

            # XXX the only reason we're doing this here is that we're afraid to
            # add a whenDeleted=CASCADE to powerups because it's inefficient,
            # however, this is arguably the archetypical use of
            # whenDeleted=CASCADE.  Soon we need to figure out a real solution
            # (but I have no idea what it is). -glyph

            for iface in sig.store.interfacesFor(sig):
                sig.store.powerDown(sig, iface)
            tracker.deleteFromStore()
            sig.deleteFromStore()