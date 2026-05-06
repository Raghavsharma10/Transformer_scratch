def _call(self, cmd, get_output):
        """Calls a command through the SSH connection.

        Remote stderr gets printed to this program's stderr. Output is captured
        and may be returned.
        """
        server_err = self.server_logger()

        chan = self.get_client().get_transport().open_session()
        try:
            logger.debug("Invoking %r%s",
                         cmd, " (stdout)" if get_output else "")
            chan.exec_command('/bin/sh -c %s' % shell_escape(cmd))
            output = b''
            while True:
                r, w, e = select.select([chan], [], [])
                if chan not in r:
                    continue  # pragma: no cover
                recvd = False
                while chan.recv_stderr_ready():
                    data = chan.recv_stderr(1024)
                    server_err.append(data)
                    recvd = True
                while chan.recv_ready():
                    data = chan.recv(1024)
                    if get_output:
                        output += data
                    recvd = True
                if not recvd and chan.exit_status_ready():
                    break
            output = output.rstrip(b'\r\n')
            return chan.recv_exit_status(), output
        finally:
            server_err.done()
            chan.close()