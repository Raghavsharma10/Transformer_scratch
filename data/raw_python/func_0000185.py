def delete(self, dn: str) -> None:
        """
        delete a dn in the ldap database; see ldap module. doesn't return a
        result if transactions enabled.
        """

        _debug("delete", self)

        # get copy of cache
        result = self._cache_get_for_dn(dn)

        # remove special values that can't be added
        def delete_attribute(name):
            if name in result:
                del result[name]
        delete_attribute('entryUUID')
        delete_attribute('structuralObjectClass')
        delete_attribute('modifiersName')
        delete_attribute('subschemaSubentry')
        delete_attribute('entryDN')
        delete_attribute('modifyTimestamp')
        delete_attribute('entryCSN')
        delete_attribute('createTimestamp')
        delete_attribute('creatorsName')
        delete_attribute('hasSubordinates')
        delete_attribute('pwdFailureTime')
        delete_attribute('pwdChangedTime')
        # turn into mod_list list.
        mod_list = tldap.modlist.addModlist(result)

        _debug("revlist:", mod_list)

        # on commit carry out action; on rollback restore cached state
        def on_commit(obj):
            obj.delete(dn)

        def on_rollback(obj):
            obj.add(dn, None, mod_list)

        return self._process(on_commit, on_rollback)