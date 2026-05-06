def add(self, dn: str, mod_list: dict) -> None:
        """
        Add a DN to the LDAP database; See ldap module. Doesn't return a result
        if transactions enabled.
        """

        return self._do_with_retry(lambda obj: obj.add_s(dn, mod_list))