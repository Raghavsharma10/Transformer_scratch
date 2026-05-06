def _id(self):
        r"""
        The `SSHKey`'s ``id`` field, or if that is not defined, its
        ``fingerprint`` field.  If neither field is defined, accessing this
        attribute raises a `TypeError`.
        """
        if self.get("id") is not None:
            return self.id
        elif self.get("fingerprint") is not None:
            return self.fingerprint
        else:
            raise TypeError('SSHKey has neither .id nor .fingerprint')