def GetMemSharedSavedMB(self):
        '''Retrieves the estimated amount of physical memory on the host saved
           from copy-on-write (COW) shared guest physical memory.'''
        counter = c_uint()
        ret = vmGuestLib.VMGuestLib_GetMemSharedSavedMB(self.handle.value, byref(counter))
        if ret != VMGUESTLIB_ERROR_SUCCESS: raise VMGuestLibException(ret)
        return counter.value