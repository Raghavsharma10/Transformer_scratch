def _reset_i(self, i):
        """
            reset i-th progress information
        """
        self.count[i].value=0
        log.debug("reset counter %s", i)
        self.lock[i].acquire()
        for x in range(self.q[i].qsize()):
            self.q[i].get()
        
        self.lock[i].release()
        self.start_time[i].value = time.time()