def load_slice(self, state, start, end):
        """
        Return the memory objects overlapping with the provided slice.
        :param start: the start address
        :param end: the end address (non-inclusive)
        :returns: tuples of (starting_addr, memory_object)
        """
        items = []
        if start > self._page_addr + self._page_size or end < self._page_addr:
            l.warning("Calling load_slice on the wrong page.")
            return items

        for addr in range(max(start, self._page_addr), min(
                end, self._page_addr + self._page_size)):
            i = addr - self._page_addr
            mo = self._storage[i]
            if mo is None and hasattr(self, "from_dbg"):
                byte_val = get_debugger().get_byte(addr)
                mo = SimMemoryObject(claripy.BVV(byte_val, 8), addr)
                self._storage[i] = mo
            if mo is not None and (not items or items[-1][1] is not mo):
                items.append((addr, mo))
        #print filter(lambda x: x != None, self._storage)
        return items