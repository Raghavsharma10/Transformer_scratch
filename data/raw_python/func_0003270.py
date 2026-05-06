def syscall(self, func):
        '''
        Call the func in core context (main loop).
        
        func should like::
        
            def syscall_sample(scheduler, processor):
                something...

        where processor is a function which accept an event. When calling processor, scheduler directly process this event without
        sending it to queue.
        
        An event matcher is returned to the caller, and the caller should wait for the event immediately to get the return value from the system call.
        The SyscallReturnEvent will have 'retvalue' as the return value, or 'exception' as the exception thrown: (type, value, traceback)
        
        :param func: syscall function
        
        :returns: an event matcher to wait for the SyscallReturnEvent. If None is returned, a syscall is already scheduled;
                  return to core context at first.
        
        '''
        if getattr(self, 'syscallfunc', None) is not None:
            return None
        self.syscallfunc = func
        self.syscallmatcher = SyscallReturnEvent.createMatcher()
        return self.syscallmatcher