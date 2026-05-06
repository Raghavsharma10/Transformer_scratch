def get_ec_index(self, ec_handle):
        '''Get the index of the execution context with the given handle.

        @param ec_handle The handle of the execution context to look for.
        @type ec_handle str
        @return The index into the owned + participated arrays, suitable for
        use in methods such as @ref activate_in_ec, or -1 if the EC was not
        found.
        @raises NoECWithHandleError

        '''
        with self._mutex:
            for ii, ec in enumerate(self.owned_ecs):
                if ec.handle == ec_handle:
                    return ii
            for ii, ec in enumerate(self.participating_ecs):
                if ec.handle == ec_handle:
                    return ii + len(self.owned_ecs)
            raise exceptions.NoECWithHandleError