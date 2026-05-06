def unwind(self, values, backend, **kwargs):
        '''Unwind expression by applying *values* to the abstract nodes.

        The ``kwargs`` dictionary can contain data which can be used
        to override values
        '''
        if not hasattr(self, "_unwind_value"):
            self._unwind_value = self._unwind(values, backend, **kwargs)
        return self._unwind_value