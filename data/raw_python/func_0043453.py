def exec_command(self, command, sudo=False, **kwargs):
        """Wrapper to paramiko.SSHClient.exec_command
        """
        channel = self.client.get_transport().open_session()
        # stdin = channel.makefile('wb')
        stdout = channel.makefile('rb')
        stderr = channel.makefile_stderr('rb')

        if sudo:
            command = 'sudo -S bash -c \'%s\'' % command
        else:
            command = 'bash -c \'%s\'' % command

        logger.debug("Running command %s on '%s'", command, self.host)
        channel.exec_command(command, **kwargs)

        while not (channel.recv_ready() or channel.closed or
                   channel.exit_status_ready()):
            time.sleep(.2)

        ret = {'stdout': stdout.read().strip(), 'stderr': stderr.read().strip(),
               'exit_code': channel.recv_exit_status()}
        return ret