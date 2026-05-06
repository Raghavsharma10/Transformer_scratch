def owner_name(self):
        '''The name of the RTObject that owns this context.'''
        with self._mutex:
            if self._owner:
                return self._owner.get_component_profile().instance_name
            else:
                return ''