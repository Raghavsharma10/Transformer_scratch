def _attachChild(self, child):
        "attach a child database, returning an identifier for it"
        self._childCounter += 1
        databaseName = 'child_db_%d' % (self._childCounter,)
        self._attachedChildren[databaseName] = child
        # ATTACH DATABASE statements can't use bind paramaters, blech.
        self.executeSQL("ATTACH DATABASE '%s' AS %s" % (
                child.dbdir.child('db.sqlite').path,
                databaseName,))
        return databaseName