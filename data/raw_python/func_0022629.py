def implement(self, implementation, for_type=None, for_types=None):
        """Registers an implementing function for for_type.

        Arguments:
            implementation: Callable implementation for this type.
            for_type: The type this implementation applies to.
            for_types: Same as for_type, but takes a tuple of types.

            for_type and for_types cannot both be passed (for obvious reasons.)

        Raises:
            ValueError
        """
        unbound_implementation = self.__get_unbound_function(implementation)
        for_types = self.__get_types(for_type, for_types)

        for t in for_types:
            self._write_lock.acquire()
            try:
                self.implementations.append((t, unbound_implementation))
            finally:
                self._write_lock.release()