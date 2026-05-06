def GetMemReservationMB(self):
        '''Retrieves the minimum amount of memory that is reserved for the virtual
           machine. For information about setting a memory reservation, see "Limits
           and Reservations" on page 14.'''
        counter = c_uint()
        ret = vmGuestLib.VMGuestLib_GetMemReservationMB(self.handle.value, byref(counter))
        if ret != VMGUESTLIB_ERROR_SUCCESS: raise VMGuestLibException(ret)
        return counter.value