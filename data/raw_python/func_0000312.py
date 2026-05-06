def delete(self, dn: str) -> None:
        """
        delete a dn in the ldap database; see ldap module. doesn't return a
        result if transactions enabled.
        """

        return self._do_with_retry(lambda obj: obj.delete_s(dn))