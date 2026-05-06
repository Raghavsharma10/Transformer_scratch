def remove_from_groups(self, groups=None, all_groups=False, group_type=None):
        """
        Remove user from some PLC groups, or all of them.
        :param groups: list of group names the user should be removed from
        :param all_groups: a boolean meaning remove from all (don't specify groups or group_type in this case)
        :param group_type: the type of group (defaults to "product")
        :return: the User, so you can do User(...).remove_from_groups(...).add_role(...)
        """
        if all_groups:
            if groups or group_type:
                raise ArgumentError("When removing from all groups, do not specify specific groups or types")
            glist = "all"
        else:
            if not groups:
                raise ArgumentError("You must specify groups from which to remove the user")
            if not group_type:
                group_type = GroupTypes.product
            elif group_type in GroupTypes.__members__:
                group_type = GroupTypes[group_type]
            if group_type not in GroupTypes:
                raise ArgumentError("You must specify a GroupType value for argument group_type")
            glist = {group_type.name: [group for group in groups]}
        return self.append(remove=glist)