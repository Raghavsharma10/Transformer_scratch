def GetCpuShares(self):
        '''Retrieves the number of CPU shares allocated to the virtual machine. For
           information about how an ESX server uses CPU shares to manage virtual
           machine priority, see the vSphere Resource Management Guide.'''
        counter = c_uint()
        ret = vmGuestLib.VMGuestLib_GetCpuShares(self.handle.value, byref(counter))
        if ret != VMGUESTLIB_ERROR_SUCCESS: raise VMGuestLibException(ret)
        return counter.value