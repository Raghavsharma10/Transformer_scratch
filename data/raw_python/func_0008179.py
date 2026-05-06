def add_to_groups(self, groups=None, all_groups=False, group_type=None):
        """
        Add user to some (typically PLC) groups.  Note that, if you add to no groups, the effect
        is simply to do an "add to organization Everybody group", so we let that be done.
        :param groups: list of group names the user should be added to
        :param all_groups: a boolean meaning add to all (don't specify groups or group_type in this case)
        :param group_type: the type of group (defaults to "product")
        :return: the User, so you can do User(...).add_to_groups(...).add_role(...)
        """
        if all_groups:
            if groups or group_type:
                raise ArgumentError("When adding to all groups, do not specify specific groups or types")
            glist = "all"
        else:
            if not groups:
                groups = []
            if not group_type:
                group_type = GroupTypes.product
            elif group_type in GroupTypes.__members__:
                group_type = GroupTypes[group_type]
            if group_type not in GroupTypes:
                raise ArgumentError("You must specify a GroupType value for argument group_type")
            glist = {group_type.name: [group for group in groups]}
        return self.append(add=glist)