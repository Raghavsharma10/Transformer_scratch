def is_member(self, rtc):
        '''Is the given component a member of this composition?

        rtc may be a Component object or a string containing a component's
        instance name. Component objects are more reliable.

        Returns False if the given component is not a member of this
        composition.

        Raises NotCompositeError if this component is not a composition.

        '''
        if not self.is_composite:
            raise exceptions.NotCompositeError(self.name)
        members = self.organisations[0].obj.get_members()
        if type(rtc) is str:
            for m in members:
                if m.get_component_profile().instance_name == rtc:
                    return True
        else:
            for m in members:
                if m._is_equivalent(rtc.object):
                    return True
        return False