def open_umanager(self):
        """Used to open an uManager session.
        
        """

        if self.umanager_opened:
            return
        self.ser.write(self.cmd_umanager_invocation)
        # optimistic approach first: assume umanager is not invoked
        if self.read_loop(lambda x: x.endswith(self.umanager_prompt),self.timeout*self.umanager_waitcoeff):
            self.umanager_opened = True
        else:
            #if we are already in umanager, this will give us a fresh prompt
            self.ser.write(self.cr)
            if self.read_loop(lambda x: x.endswith(self.umanager_prompt),self.timeout):
                self.umanager_opened = True
        
        if self.umanager_opened:
            log.debug("uManager opened")
        else:
            raise Dam1021Error(1,"Failed to open uManager")