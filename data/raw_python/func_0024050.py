def GetMemTargetSizeMB(self):
        '''Retrieves the size of the target memory allocation for this virtual machine.'''
        counter = c_uint()
        ret = vmGuestLib.VMGuestLib_GetMemTargetSizeMB(self.handle.value, byref(counter))
        if ret != VMGUESTLIB_ERROR_SUCCESS: raise VMGuestLibException(ret)
        return counter.value