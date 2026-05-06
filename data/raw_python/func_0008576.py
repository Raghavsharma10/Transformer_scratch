def add_group(self, group_id, name, desc):
        '''Add a new parameter group.

        Parameters
        ----------
        group_id : int
            The numeric ID for a group to check or create.
        name : str, optional
            If a group is created, assign this name to the group.
        desc : str, optional
            If a group is created, assign this description to the group.

        Returns
        -------
        group : :class:`Group`
            A group with the given ID, name, and description.

        Raises
        ------
        KeyError
            If a group with a duplicate ID or name already exists.
        '''
        if group_id in self.groups:
            raise KeyError(group_id)
        name = name.upper()
        if name in self.groups:
            raise KeyError(name)
        group = self.groups[name] = self.groups[group_id] = Group(name, desc)
        return group