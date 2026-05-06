def OpenHandle(self):
        '''Gets a handle for use with other vSphere Guest API functions. The guest library
           handle provides a context for accessing information about the virtual machine.

           Virtual machine statistics and state data are associated with a particular guest library
           handle, so using one handle does not affect the data associated with another handle.'''
        if hasattr(self, 'handle'):
            return self.handle
        else:
            handle = c_void_p()
            ret = vmGuestLib.VMGuestLib_OpenHandle(byref(handle))
            if ret != VMGUESTLIB_ERROR_SUCCESS: raise VMGuestLibException(ret)
            return handle