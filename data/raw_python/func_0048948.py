def acquire(self, timeout=None):
        '''
            acquire - Acquire given lock. Can be blocking or nonblocking by providing a timeout.
              Returns "True" if you got the lock, otherwise "False"

            @param timeout <None/float> - Max number of seconds to wait, or None to block until we can acquire it.

            @return  <bool> - True if you got the lock, otherwise False.
        '''
        if self.held is True:
            # NOTE: Without some type of in-directory marker (like a uuid) we cannot
            #        refresh an expired lock accurately
            if os.path.exists(self.lockPath):
                return True
            # Someone removed our lock
            self.held = False

        # If we aren't going to poll at least 5 times, give us a smaller interval
        if timeout:
            if timeout / 5.0 < DEFAULT_POLL_TIME:
                pollTime = timeout / 10.0
            else:
                pollTime = DEFAULT_POLL_TIME
            
            endTime = time.time() + timeout
            keepGoing = lambda : bool(time.time() < endTime)
        else:
            pollTime = DEFAULT_POLL_TIME
            keepGoing = lambda : True

                    

        success = False
        while keepGoing():
            try:
                os.mkdir(self.lockPath) 
                success = True
                break
            except:
                time.sleep(pollTime)
                if self.maxLockAge:
                    if os.path.exists(self.lockPath) and os.stat(self.lockPath).st_mtime < time.time() - self.maxLockAge:
                        try:
                            os.rmdir(self.lockPath)
                        except:
                            # If we did not remove the lock, someone else is at the same point and contending. Let them win.
                            time.sleep(pollTime)
        
        if success is True:
            self.acquiredAt = time.time()

        self.held = success
        return success