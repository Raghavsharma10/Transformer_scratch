def installProductOn(self, userstore):
        """
        Creates an Installation in this user store for our collection
        of powerups, and then install those powerups on the user's
        store.
        """

        def install():
            i = Installation(store=userstore)
            i.types = self.types
            i.install()
        userstore.transact(install)