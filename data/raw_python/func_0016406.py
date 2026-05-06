def _read_msg(self):
        """read message from server"""

        #
        # NOTE:
        # '_recv_socket(nbytes)' was implemented as
        # 'socket.recv(nbytes, socket.MSG_WAITALL)'
        # but socket.MSG_WAITALL proved not reliable
        #

        def _recv_socket(nbytes):
            """read nbytes bytes from self.socket"""

            #
            # code below is written under the assumption that
            # 'nbytes' is smallish so that the 'while len(buf) < nbytes' loop
            # is entered rarerly
            #
            try:
                buf = self.socket.recv(nbytes)
            except IOError as err:
                raise ConnError(*err.args)

            if not buf:
                raise ShortRead(0, nbytes)

            while len(buf) < nbytes:
                try:
                    tmp = self.socket.recv(nbytes - len(buf))
                except IOError as err:
                    raise ConnError(*err.args)

                if not tmp:
                    if self.verbose:
                        print('ee', repr(buf))
                    raise ShortRead(len(buf), nbytes)

                buf += tmp

            assert len(buf) == nbytes, (buf, len(buf), nbytes)
            return buf

        data = _recv_socket(_FromServerHeader.header_size)
        header = _FromServerHeader(data)
        if self.verbose:
            print('<-', repr(header))

        # error conditions
        if header.version != 0:
            raise MalformedHeader('bad version', header)
        if header.payload > MAX_PAYLOAD:
            raise MalformedHeader('huge payload, unwilling to read', header)

        if header.payload > 0:
            payload = _recv_socket(header.payload)
            if self.verbose:
                print('..', repr(payload))
            assert header.size <= header.payload
            payload = payload[:header.size]
        else:
            payload = bytes()
        return header, payload