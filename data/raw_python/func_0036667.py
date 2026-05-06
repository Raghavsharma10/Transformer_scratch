def add_members(self, rtcs):
        '''Add other RT Components to this composite component as members.

        This component must be a composite component.

        '''
        if not self.is_composite:
            raise exceptions.NotCompositeError(self.name)
        for rtc in rtcs:
            if self.is_member(rtc):
                raise exceptions.AlreadyInCompositionError(self.name, rtc.instance_name)
        org = self.organisations[0].obj
        org.add_members([x.object for x in rtcs])
        # Force a reparse of the member information
        self._orgs = []