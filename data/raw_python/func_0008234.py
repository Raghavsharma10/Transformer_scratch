def run(self):
        """
        Version of run that traps Exceptions and stores
        them in the fifo
        """
        try:
            threading.Thread.run(self)
        except Exception:
            t, v, tb = sys.exc_info()
            error = traceback.format_exception_only(t, v)[0][:-1]
            tback = (self.name + ' Traceback (most recent call last):\n' +
                     ''.join(traceback.format_tb(tb)))
            self.fifo.put((self.name, error, tback))