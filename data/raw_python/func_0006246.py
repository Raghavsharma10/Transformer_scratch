def send_command(self, cmd):
        """
            Send a command to the remote SSH server.

        :param cmd: The command to send
        """
        logger.debug('Sending {0} command.'.format(cmd))
        self.comm_chan.sendall(cmd + '\n')