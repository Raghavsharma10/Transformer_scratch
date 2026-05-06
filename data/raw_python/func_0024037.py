def GetMemBalloonedMB(self):
        '''Retrieves the amount of memory that has been reclaimed from this virtual
           machine by the vSphere memory balloon driver (also referred to as the
           "vmmemctl" driver).'''
        counter = c_uint()
        ret = vmGuestLib.VMGuestLib_GetMemBalloonedMB(self.handle.value, byref(counter))
        if ret != VMGUESTLIB_ERROR_SUCCESS: raise VMGuestLibException(ret)
        return counter.value