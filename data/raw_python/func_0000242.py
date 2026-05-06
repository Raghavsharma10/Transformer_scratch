def rename(self, dn: str, new_rdn: str, new_base_dn: Optional[str] = None) -> None:
        """
        rename a dn in the ldap database; see ldap module. doesn't return a
        result if transactions enabled.
        """
        raise NotImplementedError()