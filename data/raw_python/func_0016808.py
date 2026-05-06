def load(self, *objs) -> "ReadTransaction":
        """
        Add one or more objects to be loaded in this transaction.

        At most 10 items can be loaded in the same transaction.  All objects will be loaded each time you
        call commit().


        :param objs: Objects to add to the set that are loaded in this transaction.
        :return: this transaction for chaining
        :raises bloop.exceptions.MissingObjects: if one or more objects aren't loaded.
        """
        self._extend([TxItem.new("get", obj) for obj in objs])
        return self