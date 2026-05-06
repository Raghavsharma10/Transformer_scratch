def GetHostProcessorSpeed(self):
        '''Retrieves the speed of the ESX system's physical CPU in MHz.'''
        counter = c_uint()
        ret = vmGuestLib.VMGuestLib_GetHostProcessorSpeed(self.handle.value, byref(counter))
        if ret != VMGUESTLIB_ERROR_SUCCESS: raise VMGuestLibException(ret)
        return counter.value