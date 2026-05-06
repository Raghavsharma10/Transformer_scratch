def migrateDown(self):
        """
        Assuming that self.avatars is a SubStore which should contain *only*
        the LoginAccount for the user I represent, remove all LoginAccounts and
        LoginMethods from that store and copy all methods from the site store
        down into it.
        """
        ss = self.avatars.open()
        def _():
            oldAccounts = ss.query(LoginAccount)
            oldMethods = ss.query(LoginMethod)
            for x in list(oldAccounts) + list(oldMethods):
                x.deleteFromStore()
            self.cloneInto(ss, ss)
            IScheduler(ss).migrateDown()
        ss.transact(_)