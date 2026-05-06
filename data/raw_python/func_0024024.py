def GetElapsedMs(self):
        '''Retrieves the number of milliseconds that have passed in the virtual
           machine since it last started running on the server. The count of elapsed
           time restarts each time the virtual machine is powered on, resumed, or
           migrated using VMotion. This value counts milliseconds, regardless of
           whether the virtual machine is using processing power during that time.

           You can combine this value with the CPU time used by the virtual machine
           (VMGuestLib_GetCpuUsedMs) to estimate the effective virtual machine
           CPU speed. cpuUsedMs is a subset of this value.'''
        counter = c_uint64()
        ret = vmGuestLib.VMGuestLib_GetElapsedMs(self.handle.value, byref(counter))
        if ret != VMGUESTLIB_ERROR_SUCCESS: raise VMGuestLibException(ret)
        return counter.value