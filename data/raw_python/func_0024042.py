def GetMemMappedMB(self):
        '''Retrieves the amount of memory that is allocated to the virtual machine.
           Memory that is ballooned, swapped, or has never been accessed is
           excluded.'''
        counter = c_uint()
        ret = vmGuestLib.VMGuestLib_GetMemMappedMB(self.handle.value, byref(counter))
        if ret != VMGUESTLIB_ERROR_SUCCESS: raise VMGuestLibException(ret)
        return counter.value