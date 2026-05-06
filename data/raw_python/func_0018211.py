def inputcooker_store_queue(self, char):
        """Put the cooked data in the input queue (with locking)"""
        self.IQUEUELOCK.acquire()
        if type(char) in [type(()), type([]), type("")]:
            for v in char:
                self.cookedq.append(v)
        else:
            self.cookedq.append(char)
        self.IQUEUELOCK.release()