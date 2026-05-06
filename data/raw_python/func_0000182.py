def add(self, dn: str, mod_list: dict) -> None:
        """
        Add a DN to the LDAP database; See ldap module. Doesn't return a result
        if transactions enabled.
        """

        _debug("add", self, dn, mod_list)

        # if rollback of add required, delete it
        def on_commit(obj):
            obj.add(dn, None, mod_list)

        def on_rollback(obj):
            obj.delete(dn)

        # process this action
        return self._process(on_commit, on_rollback)