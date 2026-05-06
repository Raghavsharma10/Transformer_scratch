def GetSessionId(self):
        '''Retrieves the VMSessionID for the current session. Call this function after calling
           VMGuestLib_UpdateInfo. If VMGuestLib_UpdateInfo has never been called,
           VMGuestLib_GetSessionId returns VMGUESTLIB_ERROR_NO_INFO.'''
        sid = c_void_p()
        ret = vmGuestLib.VMGuestLib_GetSessionId(self.handle.value, byref(sid))
        if ret != VMGUESTLIB_ERROR_SUCCESS: raise VMGuestLibException(ret)
        return sid