def GetCpuLimitMHz(self):
        '''Retrieves the upperlimit of processor use in MHz available to the virtual
           machine. For information about setting the CPU limit, see "Limits and
           Reservations" on page 14.'''
        counter = c_uint()
        ret = vmGuestLib.VMGuestLib_GetCpuLimitMHz(self.handle.value, byref(counter))
        if ret != VMGUESTLIB_ERROR_SUCCESS: raise VMGuestLibException(ret)
        return counter.value