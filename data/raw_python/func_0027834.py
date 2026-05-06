def migrateUp(self):
        """
        Copy this LoginAccount and all associated LoginMethods from my store
        (which is assumed to be a SubStore, most likely a user store) into the
        site store which contains it.
        """
        siteStore = self.store.parent
        def _():
            # No convenience method for the following because needing to do it is
            # *rare*.  It *should* be ugly; 99% of the time if you need to do this
            # you're making a mistake. -glyph
            siteStoreSubRef = siteStore.getItemByID(self.store.idInParent)
            self.cloneInto(siteStore, siteStoreSubRef)
            IScheduler(self.store).migrateUp()
        siteStore.transact(_)