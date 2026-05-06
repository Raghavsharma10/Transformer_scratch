def send_exit_status(self, status):
        """
        Send the exit status of an executed command to the client.  (This
        really only makes sense in server mode.)  Many clients expect to
        get some sort of status code back from an executed command after
        it completes.
        
        @param status: the exit code of the process
        @type status: int
        
        @since: 1.2
        """
        # in many cases, the channel will not still be open here.
        # that's fine.
        m = Message()
        m.add_byte(chr(MSG_CHANNEL_REQUEST))
        m.add_int(self.remote_chanid)
        m.add_string('exit-status')
        m.add_boolean(False)
        m.add_int(status)
        self.transport._send_user_message(m)