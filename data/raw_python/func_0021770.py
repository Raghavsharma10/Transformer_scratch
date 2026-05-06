def compare(self, dn, attr, value):
        """
        Compare the ``attr`` of the entry ``dn`` with given ``value``.

        This is a convenience wrapper for the ldap library's ``compare``
        function that returns a boolean value instead of 1 or 0.
        """
        return self.connection.compare_s(dn, attr, value) == 1