def recv_exit_status(self):
        """
        Return the exit status from the process on the server.  This is
        mostly useful for retrieving the reults of an L{exec_command}.
        If the command hasn't finished yet, this method will wait until
        it does, or until the channel is closed.  If no exit status is
        provided by the server, -1 is returned.
        
        @return: the exit code of the process on the server.
        @rtype: int
        
        @since: 1.2
        """
        self.status_event.wait()
        assert self.status_event.isSet()
        return self.exit_status