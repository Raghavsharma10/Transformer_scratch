def run(self):
        """
        Runs the given method until cancel() is called
        """
        # The "or" part is for Python 2.6
        while not (self.finished.wait(self.interval)
                   or self.finished.is_set()):
            self.function(*self.args, **self.kwargs)