def do_thread(self, arg):
        """th(read) [threadnumber]
        Without argument, display a summary of all active threads.
        The summary prints for each thread:
           1. the thread number assigned by pdb
           2. the thread name
           3. the python thread identifier
           4. the current stack frame summary for that thread
        An asterisk '*' to the left of the pdb thread number indicates the
        current thread, a plus sign '+' indicates the thread being traced by
        pdb.

        With a pdb thread number as argument, make this thread the current
        thread. The 'where', 'up' and 'down' commands apply now to the frame
        stack of this thread. The current scope is now the frame currently
        executed by this thread at the time the command is issued and the
        'list', 'll', 'args', 'p', 'pp', 'source' and 'interact' commands are
        run in the context of that frame. Note that this frame may bear no
        relationship (for a non-deadlocked thread) to that thread's current
        activity by the time you are examining the frame.
        This command does not stop the thread.
        """
        # Import the threading module in the main interpreter to get an
        # enumeration of the main interpreter threads.
        if PY3:
            try:
                import threading
            except ImportError:
                import dummy_threading as threading
        else:
            # Do not use relative import detection to avoid the RuntimeWarning:
            # Parent module 'pdb_clone' not found while handling absolute
            # import.
            try:
                threading = __import__('threading', level=0)
            except ImportError:
                threading = __import__('dummy_threading', level=0)


        if not self.pdb_thread:
            self.pdb_thread = threading.current_thread()
        if not self.current_thread:
            self.current_thread = self.pdb_thread
        current_frames = sys._current_frames()
        tlist = sorted(threading.enumerate(), key=attrgetter('name', 'ident'))
        try:
            self._do_thread(arg, current_frames, tlist)
        finally:
            # For some reason this local must be explicitly deleted in order
            # to release the subinterpreter.
            del current_frames