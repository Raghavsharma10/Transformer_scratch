def getc(self, block=True):
        """Return one character from the input queue"""
        if not block:
            if not len(self.cookedq):
                return ''
        while not len(self.cookedq):
            time.sleep(0.05)
        self.IQUEUELOCK.acquire()
        ret = self.cookedq[0]
        self.cookedq = self.cookedq[1:]
        self.IQUEUELOCK.release()
        return ret