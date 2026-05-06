def add_role(self, groups=None, role_type=RoleTypes.admin):
        """
        Make user have a role (typically PLC admin) with respect to some PLC groups.
        :param groups: list of group names the user should have this role for
        :param role_type: the role (defaults to "admin")
        :return: the User, so you can do User(...).add_role(...).add_to_groups(...)
        """
        if not groups:
            raise ArgumentError("You must specify groups to which to add the role for this user")
        if role_type in RoleTypes.__members__:
            role_type = RoleTypes[role_type]
        if role_type not in RoleTypes:
            raise ArgumentError("You must specify a RoleType value for argument role_type")
        glist = {role_type.name: [group for group in groups]}
        return self.append(addRoles=glist)