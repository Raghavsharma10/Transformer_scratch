def modify(self, dn: str, mod_list: dict) -> None:
        """
        Modify a DN in the LDAP database; See ldap module. Doesn't return a
        result if transactions enabled.
        """

        _debug("modify", self, dn, mod_list)

        # need to work out how to reverse changes in mod_list; result in revlist
        revlist = {}

        # get the current cached attributes
        result = self._cache_get_for_dn(dn)

        # find the how to reverse mod_list (for rollback) and put result in
        # revlist. Also simulate actions on cache.
        for mod_type, l in six.iteritems(mod_list):
            for mod_op, mod_vals in l:

                _debug("attribute:", mod_type)
                if mod_type in result:
                    _debug("attribute cache:", result[mod_type])
                else:
                    _debug("attribute cache is empty")
                _debug("attribute modify:", (mod_op, mod_vals))

                if mod_vals is not None:
                    if not isinstance(mod_vals, list):
                        mod_vals = [mod_vals]

                if mod_op == ldap3.MODIFY_ADD:
                    # reverse of MODIFY_ADD is MODIFY_DELETE
                    reverse = (ldap3.MODIFY_DELETE, mod_vals)

                elif mod_op == ldap3.MODIFY_DELETE and len(mod_vals) > 0:
                    # Reverse of MODIFY_DELETE is MODIFY_ADD, but only if value
                    # is given if mod_vals is None, this means all values where
                    # deleted.
                    reverse = (ldap3.MODIFY_ADD, mod_vals)

                elif mod_op == ldap3.MODIFY_DELETE \
                        or mod_op == ldap3.MODIFY_REPLACE:
                    if mod_type in result:
                        # If MODIFY_DELETE with no values or MODIFY_REPLACE
                        # then we have to replace all attributes with cached
                        # state
                        reverse = (
                            ldap3.MODIFY_REPLACE,
                            tldap.modlist.escape_list(result[mod_type])
                        )
                    else:
                        # except if we have no cached state for this DN, in
                        # which case we delete it.
                        reverse = (ldap3.MODIFY_DELETE, [])

                else:
                    raise RuntimeError("mod_op of %d not supported" % mod_op)

                reverse = [reverse]
                _debug("attribute reverse:", reverse)
                if mod_type in result:
                    _debug("attribute cache:", result[mod_type])
                else:
                    _debug("attribute cache is empty")

                revlist[mod_type] = reverse

        _debug("--")
        _debug("mod_list:", mod_list)
        _debug("revlist:", revlist)
        _debug("--")

        # now the hard stuff is over, we get to the easy stuff
        def on_commit(obj):
            obj.modify(dn, mod_list)

        def on_rollback(obj):
            obj.modify(dn, revlist)

        return self._process(on_commit, on_rollback)