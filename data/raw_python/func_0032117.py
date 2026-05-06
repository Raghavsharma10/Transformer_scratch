def removeProductFrom(self, userstore):
        """
        Uninstall all the powerups this product references and remove
        the Installation item from the user's store. Doesn't remove
        the actual powerups currently, but /should/ reactivate them if
        this product is reinstalled.
        """
        def uninstall():
            #this is probably highly insufficient, but i don't know the
            #requirements
            i = userstore.findFirst(Installation,
                                    Installation.types == self.types)
            i.uninstall()
            i.deleteFromStore()
        userstore.transact(uninstall)