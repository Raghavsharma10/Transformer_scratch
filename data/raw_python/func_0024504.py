def get_user_switchable_roles(self):
        """
        Returns user's role list except current role as a tuple
        (role.key, role.name)

        Returns:
            (list): list of tuples, user's role list except current role

        """
        roles = []
        for rs in self.current.user.role_set:
            # rs.role != self.current.role is not True after python version 2.7.12
            if rs.role.key != self.current.role.key:
                roles.append((rs.role.key, '%s %s' % (rs.role.unit.name,
                                                      rs.role.abstract_role.name)))
        return roles