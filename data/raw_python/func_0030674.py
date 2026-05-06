def load_mo(self, state, page_idx):
        """
        Loads a memory object from memory.
        :param page_idx: the index into the page
        :returns: a tuple of the object
        """
        mo = self._storage[page_idx - self._page_addr]
        #print filter(lambda x: x != None, self._storage)
        if mo is None and hasattr(self, "from_dbg"):
            byte_val = get_debugger().get_byte(page_idx)
            mo = SimMemoryObject(claripy.BVV(byte_val, 8), page_idx)
            self._storage[page_idx - self._page_addr] = mo
        return mo