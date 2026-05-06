def perform(self, cmd, msg='', extra_headers=None):
        """Perform the call"""
        tries = 0
        while 1:
            conn = None
            try:
                conn = self.get_connection()
                if hasattr(msg, 'read') and hasattr(msg, 'fileno'):
                    msg_length = str(os.fstat(msg.fileno()).st_size)
                elif hasattr(msg, 'read'):
                    msg.seek(0, 2)
                    msg_length = str(msg.tell() + 2)
                else:
                    if msg:
                        try:
                            msg_length = str(len(msg) + 2)
                        except TypeError:
                            conn.close()
                            raise ValueError(
                                'msg param should be a string or file handle')
                    else:
                        msg_length = '2'

                headers = self.get_headers(cmd, msg_length, extra_headers)

                if isinstance(msg, types.StringTypes):
                    if self.gzip and msg:
                        msg = compress(msg + '\r\n', self.compress_level)
                    else:
                        msg = msg + '\r\n'
                    conn.send(headers + msg)
                else:
                    conn.send(headers)
                    if hasattr(msg, 'read'):
                        if hasattr(msg, 'seek'):
                            msg.seek(0)
                        conn.sendfile(msg, self.gzip, self.compress_level)
                conn.send('\r\n')
                try:
                    conn.socket().shutdown(socket.SHUT_WR)
                except socket.error:
                    pass
                return get_response(cmd, conn)
            except socket.gaierror as err:
                if conn is not None:
                    conn.release()
                raise SpamCError(str(err))
            except socket.timeout as err:
                if conn is not None:
                    conn.release()
                raise SpamCTimeOutError(str(err))
            except socket.error as err:
                if conn is not None:
                    conn.close()
                errors = (errno.EAGAIN, errno.EPIPE, errno.EBADF,
                          errno.ECONNRESET)
                if err[0] not in errors or tries >= self.max_tries:
                    raise SpamCError("socket.error: %s" % str(err))
            except BaseException:
                if conn is not None:
                    conn.release()
                raise
            tries += 1
            self.backend_mod.sleep(self.wait_tries)