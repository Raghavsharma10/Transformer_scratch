def reset_in_ec(self, ec_index):
        '''Reset this component in an execution context.

        @param ec_index The index of the execution context to reset in. This
                        index is into the total array of contexts, that is both
                        owned and participating contexts. If the value of
                        ec_index is greater than the length of @ref owned_ecs,
                        that length is subtracted from ec_index and the result
                        used as an index into @ref participating_ecs.

        '''
        with self._mutex:
            if ec_index >= len(self.owned_ecs):
                ec_index -= len(self.owned_ecs)
                if ec_index >= len(self.participating_ecs):
                    raise exceptions.BadECIndexError(ec_index)
                ec = self.participating_ecs[ec_index]
            else:
                ec = self.owned_ecs[ec_index]
            ec.reset_component(self._obj)