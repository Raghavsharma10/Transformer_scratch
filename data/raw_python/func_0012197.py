def start(self):
        '''Equivalent to `run`, but instead of blocking the current thread,
        return a WaitHandle that doesn't block until `wait` is called. This is
        currently implemented with a simple background thread, though in theory
        it could avoid using threads in most cases.'''
        thread = ThreadWithReturn(self.run)
        thread.start()
        return WaitHandle(thread)