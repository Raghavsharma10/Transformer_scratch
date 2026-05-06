def GetMemSwappedMB(self):
        '''Retrieves the amount of memory that has been reclaimed from this virtual
           machine by transparently swapping guest memory to disk.'''
        counter = c_uint()
        ret = vmGuestLib.VMGuestLib_GetMemSwappedMB(self.handle.value, byref(counter))
        if ret != VMGUESTLIB_ERROR_SUCCESS: raise VMGuestLibException(ret)
        return counter.value