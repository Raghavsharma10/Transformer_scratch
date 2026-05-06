def _find_and_cache_best_function(self, dispatch_type):
        """Finds the best implementation of this function given a type.

        This function caches the result, and uses locking for thread safety.

        Returns:
            Implementing function, in below order of preference:
            1. Explicitly registered implementations (through
               multimethod.implement) for types that 'dispatch_type' either is
               or inherits from directly.
            2. Explicitly registered implementations accepting an abstract type
               (interface) in which dispatch_type participates (through
               abstract_type.register() or the convenience methods).
            3. Default behavior of the multimethod function. This will usually
               raise a NotImplementedError, by convention.

        Raises:
            TypeError: If two implementing functions are registered for
                different abstract types, and 'dispatch_type' participates in
                both, and no order of preference was specified using
                prefer_type.
        """
        result = self._dispatch_table.get(dispatch_type)
        if result:
            return result

        # The outer try ensures the lock is always released.
        with self._write_lock:
            try:
                dispatch_mro = dispatch_type.mro()
            except TypeError:
                # Not every type has an MRO.
                dispatch_mro = ()

            best_match = None
            result_type = None

            for candidate_type, candidate_func in self.implementations:
                if not issubclass(dispatch_type, candidate_type):
                    # Skip implementations that are obviously unrelated.
                    continue

                try:
                    # The candidate implementation may be for a type that's
                    # actually in the MRO, or it may be for an abstract type.
                    match = dispatch_mro.index(candidate_type)
                except ValueError:
                    # This means we have an implementation for an abstract
                    # type, which ranks below all concrete types.
                    match = None

                if best_match is None:
                    if result and match is None:
                        # Already have a result, and no order of preference.
                        # This is probably because the type is a member of two
                        # abstract types and we have separate implementations
                        # for those two abstract types.

                        if self._preferred(candidate_type, over=result_type):
                            result = candidate_func
                            result_type = candidate_type
                        elif self._preferred(result_type, over=candidate_type):
                            # No need to update anything.
                            pass
                        else:
                            raise TypeError(
                                "Two candidate implementations found for "
                                "multimethod function %s (dispatch type %s) "
                                "and neither is preferred." %
                                (self.func_name, dispatch_type))
                    else:
                        result = candidate_func
                        result_type = candidate_type
                        best_match = match

                if (match or 0) < (best_match or 0):
                    result = candidate_func
                    result_type = candidate_type
                    best_match = match

            self._dispatch_table[dispatch_type] = result
            return result