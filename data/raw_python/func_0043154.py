def wait(self):
        """
        PUT THREAD IN WAIT STATE UNTIL SIGNAL IS ACTIVATED
        """
        if self._go:
            return True

        with self.lock:
            if self._go:
                return True
            stopper = _allocate_lock()
            stopper.acquire()
            if not self.waiting_threads:
                self.waiting_threads = [stopper]
            else:
                self.waiting_threads.append(stopper)

        DEBUG and self._name and Log.note("wait for go {{name|quote}}", name=self.name)
        stopper.acquire()
        DEBUG and self._name and Log.note("GOing! {{name|quote}}", name=self.name)
        return True