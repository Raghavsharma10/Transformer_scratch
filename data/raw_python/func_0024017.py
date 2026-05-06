def UpdateInfo(self):
        '''Updates information about the virtual machine. This information is associated with
           the VMGuestLibHandle.

           VMGuestLib_UpdateInfo requires similar CPU resources to a system call and
           therefore can affect performance. If you are concerned about performance, minimize
           the number of calls to VMGuestLib_UpdateInfo.

           If your program uses multiple threads, each thread must use a different handle.
           Otherwise, you must implement a locking scheme around update calls. The vSphere
           Guest API does not implement internal locking around access with a handle.'''
        ret = vmGuestLib.VMGuestLib_UpdateInfo(self.handle.value)
        if ret != VMGUESTLIB_ERROR_SUCCESS: raise VMGuestLibException(ret)