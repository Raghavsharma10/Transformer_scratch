def CloseHandle(self):
        '''Releases a handle acquired with VMGuestLib_OpenHandle'''
        if hasattr(self, 'handle'):
            ret = vmGuestLib.VMGuestLib_CloseHandle(self.handle.value)
            if ret != VMGUESTLIB_ERROR_SUCCESS: raise VMGuestLibException(ret)
            del(self.handle)