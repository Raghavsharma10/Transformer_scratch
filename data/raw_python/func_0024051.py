def GetMemUsedMB(self):
        '''Retrieves the estimated amount of physical host memory currently
           consumed for this virtual machine's physical memory.'''
        counter = c_uint()
        ret = vmGuestLib.VMGuestLib_GetMemUsedMB(self.handle.value, byref(counter))
        if ret != VMGUESTLIB_ERROR_SUCCESS: raise VMGuestLibException(ret)
        return counter.value