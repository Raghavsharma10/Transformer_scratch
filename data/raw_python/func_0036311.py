def participant_names(self):
        '''The names of the RTObjects participating in this context.'''
        with self._mutex:
            return [obj.get_component_profile().instance_name \
                    for obj in self._participants]