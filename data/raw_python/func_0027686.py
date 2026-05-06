def call(self, itemMethod):
        """
        Invoke the given bound item method in the batch process.

        Return a Deferred which fires when the method has been invoked.
        """
        item = itemMethod.im_self
        method = itemMethod.im_func.func_name
        return self.batchController.getProcess().addCallback(
            CallItemMethod(storepath=item.store.dbdir,
                           storeid=item.storeID,
                           method=method).do)