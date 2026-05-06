def remove_role(self, groups=None, role_type=RoleTypes.admin):
        """
        Remove user from a role (typically admin) of some groups.
        :param groups: list of group names the user should NOT have this role for
        :param role_type: the type of role (defaults to "admin")
        :return: the User, so you can do User(...).remove_role(...).remove_from_groups(...)
        """
        if not groups:
            raise ArgumentError("You must specify groups from which to remove the role for this user")
        if role_type in RoleTypes.__members__:
            role_type = RoleTypes[role_type]
        if role_type not in RoleTypes:
            raise ArgumentError("You must specify a RoleType value for argument role_type")
        glist = {role_type.name: [group for group in groups]}
        return self.append(removeRoles=glist)