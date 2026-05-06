def allRoles(self, memo=None):
        """
        Identify all the roles that this role is authorized to act as.

        @param memo: used only for recursion.  Do not pass this.

        @return: an iterator of all roles that this role is a member of,
        including itself.
        """
        if memo is None:
            memo = set()
        elif self in memo:
            # this is bad, but we have successfully detected and prevented the
            # only really bad symptom, an infinite loop.
            return
        memo.add(self)
        yield self
        for groupRole in self.store.query(Role,
                                          AND(RoleRelationship.member == self,
                                              RoleRelationship.group == Role.storeID)):
            for roleRole in groupRole.allRoles(memo):
                yield roleRole