def migrateDown(self):
        """
        Remove the components in the site store for this SubScheduler.
        """
        subStore = self.store.parent.getItemByID(self.store.idInParent)
        ssph = self.store.parent.findUnique(
            _SubSchedulerParentHook,
            _SubSchedulerParentHook.subStore == subStore,
            default=None)
        if ssph is not None:
            te = self.store.parent.findUnique(TimedEvent,
                                              TimedEvent.runnable == ssph,
                                              default=None)
            if te is not None:
                te.deleteFromStore()
            ssph.deleteFromStore()