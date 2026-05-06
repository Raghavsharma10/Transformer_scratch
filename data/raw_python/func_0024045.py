def GetMemSharedMB(self):
        '''Retrieves the amount of physical memory associated with this virtual
           machine that is copy-on-write (COW) shared on the host.'''
        counter = c_uint()
        ret = vmGuestLib.VMGuestLib_GetMemSharedMB(self.handle.value, byref(counter))
        if ret != VMGUESTLIB_ERROR_SUCCESS: raise VMGuestLibException(ret)
        return counter.value