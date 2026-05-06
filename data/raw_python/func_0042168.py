def stop(self):
        """
        BLOCKS UNTIL ALL THREADS HAVE STOPPED
        THEN RUNS sys.exit(0)
        """
        global DEBUG

        self_thread = Thread.current()
        if self_thread != MAIN_THREAD or self_thread != self:
            Log.error("Only the main thread can call stop() on main thread")

        DEBUG = True
        self.please_stop.go()

        join_errors = []
        with self.child_lock:
            children = copy(self.children)
        for c in reversed(children):
            DEBUG and c.name and Log.note("Stopping thread {{name|quote}}", name=c.name)
            try:
                c.stop()
            except Exception as e:
                join_errors.append(e)

        for c in children:
            DEBUG and c.name and Log.note("Joining on thread {{name|quote}}", name=c.name)
            try:
                c.join()
            except Exception as e:
                join_errors.append(e)

            DEBUG and c.name and Log.note("Done join on thread {{name|quote}}", name=c.name)

        if join_errors:
            Log.error("Problem while stopping {{name|quote}}", name=self.name, cause=unwraplist(join_errors))

        self.stop_logging()
        self.timers.stop()
        self.timers.join()

        write_profiles(self.cprofiler)
        DEBUG and Log.note("Thread {{name|quote}} now stopped", name=self.name)
        sys.exit()