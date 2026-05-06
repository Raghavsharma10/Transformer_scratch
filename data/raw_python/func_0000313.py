def rename(self, dn: str, new_rdn: str, new_base_dn: Optional[str] = None) -> None:
        """
        rename a dn in the ldap database; see ldap module. doesn't return a
        result if transactions enabled.
        """

        return self._do_with_retry(
            lambda obj: obj.rename_s(dn, new_rdn, new_base_dn))