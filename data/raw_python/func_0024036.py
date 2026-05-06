def GetMemActiveMB(self):
        '''Retrieves the amount of memory the virtual machine is actively using its
           estimated working set size.'''
        counter = c_uint()
        ret = vmGuestLib.VMGuestLib_GetMemActiveMB(self.handle.value, byref(counter))
        if ret != VMGUESTLIB_ERROR_SUCCESS: raise VMGuestLibException(ret)
        return counter.value