def state(self):
        '''The merged state of all the execution context states, which can be
        used as the overall state of this component.

        The order of precedence is:
            Error > Active > Inactive > Created > Unknown

        '''
        def merge_state(current, new):
            if new == self.ERROR:
                return self.ERROR
            elif new == self.ACTIVE and current != self.ERROR:
                return self.ACTIVE
            elif new == self.INACTIVE and \
                    current not in [self.ACTIVE, self.ERROR]:
                return self.INACTIVE
            elif new == self.CREATED and \
                    current not in [self.ACTIVE, self.ERROR, self.INACTIVE]:
                return self.CREATED
            elif current not in [self.ACTIVE, self.ERROR, self.INACTIVE,
                                 self.CREATED]:
                return self.UNKNOWN
            return current

        with self._mutex:
            if not self.owned_ec_states and not self.participating_ec_states:
                return self.UNKNOWN
            merged_state = self.CREATED
            if self.owned_ec_states:
                for ec_state in self.owned_ec_states:
                    merged_state = merge_state(merged_state, ec_state)
            if self.participating_ec_states:
                for ec_state in self.participating_ec_states:
                    merged_state = merge_state(merged_state, ec_state)
            return merged_state