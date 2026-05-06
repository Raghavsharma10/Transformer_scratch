def exec_command(self, command, bufsize=-1):
        """
        Execute a command on the SSH server.  A new L{Channel} is opened and
        the requested command is executed.  The command's input and output
        streams are returned as python C{file}-like objects representing
        stdin, stdout, and stderr.

        @param command: the command to execute
        @type command: str
        @param bufsize: interpreted the same way as by the built-in C{file()} function in python
        @type bufsize: int
        @return: the stdin, stdout, and stderr of the executing command
        @rtype: tuple(L{ChannelFile}, L{ChannelFile}, L{ChannelFile})

        @raise SSHException: if the server fails to execute the command
        """
        chan = self._transport.open_session()
        chan.exec_command(command)
        stdin = chan.makefile('wb', bufsize)
        stdout = chan.makefile('rb', bufsize)
        stderr = chan.makefile_stderr('rb', bufsize)
        return stdin, stdout, stderr