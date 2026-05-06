def get_ec(self, ec_handle):
        '''Get a reference to the execution context with the given handle.

        @param ec_handle The handle of the execution context to look for.
        @type ec_handle str
        @return A reference to the ExecutionContext object corresponding to
        the ec_handle.
        @raises NoECWithHandleError

        '''
        with self._mutex:
            for ec in self.owned_ecs:
                if ec.handle == ec_handle:
                    return ec
            for ec in self.participating_ecs:
                if ec.handle == ec_handle:
                    return ec
            raise exceptions.NoECWithHandleError