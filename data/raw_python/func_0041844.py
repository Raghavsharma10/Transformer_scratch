def is_member_of(self, group_name):
        """Return True if member of LDAP group, otherwise return False"""
        group_dn = 'cn=%s,cn=groups,cn=accounts,%s' % (group_name, self._base_dn)
        if str(group_dn).lower() in [str(i).lower() for i in self.member_of]:
            return True
        else:
            return False