def rename(self, dn: str, new_rdn: str, new_base_dn: Optional[str] = None) -> None:
        """
        rename a dn in the ldap database; see ldap module. doesn't return a
        result if transactions enabled.
        """

        _debug("rename", self, dn, new_rdn, new_base_dn)

        # split up the parameters
        split_dn = tldap.dn.str2dn(dn)
        split_newrdn = tldap.dn.str2dn(new_rdn)
        assert(len(split_newrdn) == 1)

        # make dn unqualified
        rdn = tldap.dn.dn2str(split_dn[0:1])

        # make newrdn fully qualified dn
        tmplist = [split_newrdn[0]]
        if new_base_dn is not None:
            tmplist.extend(tldap.dn.str2dn(new_base_dn))
            old_base_dn = tldap.dn.dn2str(split_dn[1:])
        else:
            tmplist.extend(split_dn[1:])
            old_base_dn = None
        newdn = tldap.dn.dn2str(tmplist)

        _debug("--> commit  ", self, dn, new_rdn, new_base_dn)
        _debug("--> rollback", self, newdn, rdn, old_base_dn)

        # on commit carry out action; on rollback reverse rename
        def on_commit(obj):
            obj.modify_dn(dn, new_rdn, new_superior=new_base_dn)

        def on_rollback(obj):
            obj.modify_dn(newdn, rdn, new_superior=old_base_dn)

        return self._process(on_commit, on_rollback)