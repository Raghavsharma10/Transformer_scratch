def inputcooker_store_queue(self, char):
        """Put the cooked data in the input queue (no locking needed)"""
        if type(char) in [type(()), type([]), type("")]:
            for v in char:
                self.cookedq.put(v)
        else:
            self.cookedq.put(char)