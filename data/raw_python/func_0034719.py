def close_umanager(self, force=False):
        """Used to close an uManager session.
        
        :param force: try to close a session regardless of a connection object internal state
        """
      
        if not (force or self.umanager_opened):
            return
        # make sure we've got a fresh prompt
        self.ser.write(self.cr)
        if self.read_loop(lambda x: x.endswith(self.umanager_prompt),self.timeout):
            self.ser.write(''.join((self.cmd_umanager_termination,self.cr)))
            if self.read_loop(lambda x: x.endswith(self.buf_on_exit),self.timeout):
                log.debug("uManager closed")
            else:
                raise Dam1021Error(2,"Failed to close uManager")
        else:
            log.debug("uManager already closed")
            
        self.umanager_opened = False