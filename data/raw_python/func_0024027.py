def GetHostMemMappedMB(self):
        '''Undocumented.'''
        counter = c_uint()
        ret = vmGuestLib.VMGuestLib_GetHostMemMappedMB(self.handle.value, byref(counter))
        if ret != VMGUESTLIB_ERROR_SUCCESS: raise VMGuestLibException(ret)
        return counter.value