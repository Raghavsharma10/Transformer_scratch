def get_unconstrained_bytes(self, name, bits, source=None, key=None, inspect=True, events=True, **kwargs):
        """
        Get some consecutive unconstrained bytes.

        :param name: Name of the unconstrained variable
        :param bits: Size of the unconstrained variable
        :param source: Where those bytes are read from. Currently it is only used in under-constrained symbolic
                    execution so that we can track the allocation depth.
        :return: The generated variable
        """

        if (self.category == 'mem' and
                options.CGC_ZERO_FILL_UNCONSTRAINED_MEMORY in self.state.options):
            # CGC binaries zero-fill the memory for any allocated region
            # Reference: (https://github.com/CyberGrandChallenge/libcgc/blob/master/allocate.md)
            return self.state.solver.BVV(0, bits)
        elif options.SPECIAL_MEMORY_FILL in self.state.options and self.state._special_memory_filler is not None:
            return self.state._special_memory_filler(name, bits, self.state)
        else:
            if options.UNDER_CONSTRAINED_SYMEXEC in self.state.options:
                if source is not None and type(source) is int:
                    alloc_depth = self.state.uc_manager.get_alloc_depth(source)
                    kwargs['uc_alloc_depth'] = 0 if alloc_depth is None else alloc_depth + 1
            r = self.state.solver.Unconstrained(name, bits, key=key, inspect=inspect, events=events, **kwargs)
            return r