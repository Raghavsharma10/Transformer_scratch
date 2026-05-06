def GetMemLimitMB(self):
        '''Retrieves the upper limit of memory that is available to the virtual
           machine. For information about setting a memory limit, see "Limits and
           Reservations" on page 14.'''
        counter = c_uint()
        ret = vmGuestLib.VMGuestLib_GetMemLimitMB(self.handle.value, byref(counter))
        if ret != VMGUESTLIB_ERROR_SUCCESS: raise VMGuestLibException(ret)
        return counter.value