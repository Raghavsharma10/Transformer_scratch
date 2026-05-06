def remove_members(self, rtcs):
        '''Remove other RT Components from this composite component.

        rtcs is a list of components to remove. Each element must be either an
        rtctree.Component object or a string containing a component's instance
        name. rtctree.Component objects are more reliable.

        This component must be a composite component.

        '''
        if not self.is_composite:
            raise exceptions.NotCompositeError(self.name)
        org = self.organisations[0].obj
        members = org.get_members()
        for rtc in rtcs:
            if type(rtc) == str:
                rtc_name = rtc
            else:
                rtc_name = rtc.instance_name
            # Check if the RTC actually is a member
            if not self.is_member(rtc):
                raise exceptions.NotInCompositionError(self.name, rtc_name)
            # Remove the RTC from the composition
            org.remove_member(rtc_name)
        # Force a reparse of the member information
        self._orgs = []