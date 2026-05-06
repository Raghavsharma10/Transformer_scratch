def get_security_group_id(self, name):
        """
        Take name string, give back security group ID.

        To get around VPC's API being stupid.
        """
        # Memoize entire list of groups
        if not hasattr(self, '_security_groups'):
            self._security_groups = {}
            for group in self.get_all_security_groups():
                self._security_groups[group.name] = group.id
        return self._security_groups[name]