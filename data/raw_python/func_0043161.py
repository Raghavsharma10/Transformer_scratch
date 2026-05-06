def wait(self, till=None):
        """
        THE ASSUMPTION IS wait() WILL ALWAYS RETURN WITH THE LOCK ACQUIRED
        :param till: WHEN TO GIVE UP WAITING FOR ANOTHER THREAD TO SIGNAL
        :return: True IF SIGNALED TO GO, False IF till WAS SIGNALED
        """
        waiter = Signal()
        if self.waiting:
            DEBUG and _Log.note("waiting with {{num}} others on {{name|quote}}", num=len(self.waiting), name=self.name, stack_depth=1)
            self.waiting.insert(0, waiter)
        else:
            DEBUG and _Log.note("waiting by self on {{name|quote}}", name=self.name)
            self.waiting = [waiter]

        try:
            self.lock.release()
            DEBUG and _Log.note("out of lock {{name|quote}}", name=self.name)
            (waiter | till).wait()
            if DEBUG:
                _Log.note("done minimum wait (for signal {{till|quote}})", till=till.name if till else "", name=self.name)
        except Exception as e:
            if not _Log:
                _late_import()
            _Log.warning("problem", cause=e)
        finally:
            self.lock.acquire()
            DEBUG and _Log.note("re-acquired lock {{name|quote}}", name=self.name)

        try:
            self.waiting.remove(waiter)
            DEBUG and _Log.note("removed own signal from {{name|quote}}", name=self.name)
        except Exception:
            pass

        return bool(waiter)