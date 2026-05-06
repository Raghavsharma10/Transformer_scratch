def GetCpuStolenMs(self):
        '''Retrieves the number of milliseconds that the virtual machine was in a
           ready state (able to transition to a run state), but was not scheduled to run.'''
        counter = c_uint64()
        ret = vmGuestLib.VMGuestLib_GetCpuStolenMs(self.handle.value, byref(counter))
        if ret != VMGUESTLIB_ERROR_SUCCESS: raise VMGuestLibException(ret)
        return counter.value