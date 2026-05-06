def GetMemOverheadMB(self):
        '''Retrieves the amount of "overhead" memory associated with this virtual
           machine that is currently consumed on the host system. Overhead
           memory is additional memory that is reserved for data structures required
           by the virtualization layer.'''
        counter = c_uint()
        ret = vmGuestLib.VMGuestLib_GetMemOverheadMB(self.handle.value, byref(counter))
        if ret != VMGUESTLIB_ERROR_SUCCESS: raise VMGuestLibException(ret)
        return counter.value