def join(self, till=None):
        """
        RETURN THE RESULT {"response":r, "exception":e} OF THE THREAD EXECUTION (INCLUDING EXCEPTION, IF EXISTS)
        """
        if self is Thread:
            Log.error("Thread.join() is not a valid call, use t.join()")

        with self.child_lock:
            children = copy(self.children)
        for c in children:
            c.join(till=till)

        DEBUG and Log.note("{{parent|quote}} waiting on thread {{child|quote}}", parent=Thread.current().name, child=self.name)
        (self.stopped | till).wait()
        if self.stopped:
            self.parent.remove_child(self)
            if not self.end_of_thread.exception:
                return self.end_of_thread.response
            else:
                Log.error("Thread {{name|quote}} did not end well", name=self.name, cause=self.end_of_thread.exception)
        else:
            raise Except(context=THREAD_TIMEOUT)