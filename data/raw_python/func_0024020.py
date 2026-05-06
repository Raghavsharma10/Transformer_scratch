def GetCpuReservationMHz(self):
        '''Retrieves the minimum processing power in MHz reserved for the virtual
           machine. For information about setting a CPU reservation, see "Limits and
           Reservations" on page 14.'''
        counter = c_uint()
        ret = vmGuestLib.VMGuestLib_GetCpuReservationMHz(self.handle.value, byref(counter))
        if ret != VMGUESTLIB_ERROR_SUCCESS: raise VMGuestLibException(ret)
        return counter.value