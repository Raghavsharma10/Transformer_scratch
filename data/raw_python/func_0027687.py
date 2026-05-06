def processWhileRunning(self):
        """
        Run tasks until stopService is called.
        """
        work = self.step()
        for result, more in work:
            yield result
            if not self.running:
                break
            if more:
                delay = 0.1
            else:
                delay = 10.0
            yield task.deferLater(reactor, delay, lambda: None)