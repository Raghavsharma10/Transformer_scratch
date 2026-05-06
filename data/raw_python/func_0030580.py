def copy(self, _):
        """
        Return a copy of the SimMemory.
        """
        #l.debug("Copying %d bytes of memory with id %s." % (len(self.mem), self.id))
        c = SimSymbolicDbgMemory(
            mem=self.mem.branch(),
            memory_id=self.id,
            endness=self.endness,
            abstract_backer=self._abstract_backer,
            read_strategies=[ s.copy() for s in self.read_strategies ],
            write_strategies=[ s.copy() for s in self.write_strategies ],
            stack_region_map=self._stack_region_map,
            generic_region_map=self._generic_region_map
        )

        return c