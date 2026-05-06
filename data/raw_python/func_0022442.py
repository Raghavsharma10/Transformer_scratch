def sleep(self, time):
        """
        Sleep(no action) for *time* (in millisecond)
        """
        target = 'wait for %s' % str(time)
        self.device(text=target).wait.exists(timeout=time)