def stop(self):
        """
        SEND STOP SIGNAL, DO NOT BLOCK
        """
        with self.child_lock:
            children = copy(self.children)
        for c in children:
            DEBUG and c.name and Log.note("Stopping thread {{name|quote}}", name=c.name)
            c.stop()
        self.please_stop.go()

        DEBUG and Log.note("Thread {{name|quote}} got request to stop", name=self.name)