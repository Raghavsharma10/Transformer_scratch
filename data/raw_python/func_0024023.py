def GetCpuUsedMs(self):
        '''Retrieves the number of milliseconds during which the virtual machine
           has used the CPU. This value includes the time used by the guest
           operating system and the time used by virtualization code for tasks for this
           virtual machine. You can combine this value with the elapsed time
           (VMGuestLib_GetElapsedMs) to estimate the effective virtual machine
           CPU speed. This value is a subset of elapsedMs.'''
        counter = c_uint64()
        ret = vmGuestLib.VMGuestLib_GetCpuUsedMs(self.handle.value, byref(counter))
        if ret != VMGUESTLIB_ERROR_SUCCESS: raise VMGuestLibException(ret)
        return counter.value