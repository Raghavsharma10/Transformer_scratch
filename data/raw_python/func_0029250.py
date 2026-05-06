def _send_command_raw(self, command, opt=''):
        """
        Description:

            The TV doesn't handle long running connections very well,
            so we open a new connection every time.
            There might be a better way to do this,
            but it's pretty quick and resilient.

        Returns:
            If a value is being requested ( opt2 is "?" ),
            then the return value is returned.
            If a value is being set,
            it returns True for "OK" or False for "ERR"
        """
        # According to the documentation:
        # http://files.sharpusa.com/Downloads/ForHome/
        # HomeEntertainment/LCDTVs/Manuals/tel_man_LC40_46_52_60LE830U.pdf
        # Page 58 - Communication conditions for IP
        # The connection could be lost (but not only after 3 minutes),
        # so we need to the remote commands to be sure about states
        end_time = time.time() + self.timeout
        while time.time() < end_time:
            try:
                # Connect
                sock_con = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock_con.settimeout(self.connection_timeout)
                sock_con.connect((self.ip_address, self.port))

                # Authenticate
                sock_con.send(self.auth)
                sock_con.recv(1024)
                sock_con.recv(1024)

                # Send command
                if opt != '':
                    command += str(opt)
                sock_con.send(str.encode(command.ljust(8) + '\r'))
                status = bytes.decode(sock_con.recv(1024)).strip()
            except (OSError, socket.error) as exp:
                time.sleep(0.1)
                if time.time() >= end_time:
                    raise exp
            else:
                sock_con.close()
                # Sometimes the status is empty so
                # We need to retry
                if status != u'':
                    break

        if status == "OK":
            return True
        elif status == "ERR":
            return False
        else:
            try:
                return int(status)
            except ValueError:
                return status