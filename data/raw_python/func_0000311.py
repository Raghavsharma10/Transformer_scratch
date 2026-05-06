def modify(self, dn: str, mod_list: dict) -> None:
        """
        Modify a DN in the LDAP database; See ldap module. Doesn't return a
        result if transactions enabled.
        """

        return self._do_with_retry(lambda obj: obj.modify_s(dn, mod_list))