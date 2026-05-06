def process_IAC(self, sock, cmd, option):
        """
            Read in and parse IAC commands as passed by telnetlib.

            SB/SE commands are stored in sbdataq, and passed in w/ a command
            of SE.
        """
        if cmd == DO:
            if option == TM:
                # timing mark - send WILL into outgoing stream
                os.write(self.remote, IAC + WILL + TM)
            else:
                pass
        elif cmd == IP:
            # interrupt process
            os.write(self.local, IPRESP)
        elif cmd == SB:
            pass
        elif cmd == SE:
            option = self.sbdataq[0]
            if option == NAWS[0]:
                # negotiate window size.
                cols = six.indexbytes(self.sbdataq, 1)
                rows = six.indexbytes(self.sbdataq, 2)
                s = struct.pack('HHHH', rows, cols, 0, 0)
                fcntl.ioctl(self.local, termios.TIOCSWINSZ, s)
        elif cmd == DONT:
            pass
        else:
            pass