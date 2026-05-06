def becomeMemberOf(self, groupRole):
        """
        Instruct this (user or group) Role to become a member of a group role.

        @param groupRole: The role that this group should become a member of.
        """
        self.store.findOrCreate(RoleRelationship,
                                group=groupRole,
                                member=self)