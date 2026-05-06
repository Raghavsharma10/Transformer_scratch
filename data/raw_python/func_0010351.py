def _select(self):
        """
        While the client is not marked as closed, performs a socket select
        on all PushSession sockets.  If any data is received, parses and
        forwards it on to the callback function.  If the callback is
        successful, a PublishMessageReceived message is sent.
        """
        try:
            while not self.closed:
                try:
                    inputready = select.select(self.sessions.keys(), [], [], 0.1)[0]
                    for sock in inputready:
                        session = self.sessions[sock]
                        sck = session.socket

                        if sck is None:
                            # Socket has since been deleted, continue
                            continue

                        # If no defined message length, nothing has been
                        # consumed yet, parse the header.
                        if session.message_length == 0:
                            # Read header information before receiving rest of
                            # message.
                            response_type = _read_msg_header(session)
                            if response_type == NO_DATA:
                                # No data could be read, assume socket closed.
                                if session.socket is not None:
                                    self.log.error("Socket closed for Monitor %s." % session.monitor_id)
                                    self._restart_session(session)
                                continue
                            elif response_type == INCOMPLETE:
                                # More Data to be read.  Continue.
                                continue
                            elif response_type != PUBLISH_MESSAGE:
                                self.log.warn("Response Type (%x) does not match PublishMessage (%x)"
                                              % (response_type, PUBLISH_MESSAGE))
                                continue

                        try:
                            if not _read_msg(session):
                                # Data not completely read, continue.
                                continue
                        except PushException as err:
                            # If Socket is None, it was closed,
                            # otherwise it was closed when it shouldn't
                            # have been restart it.
                            session.data = six.b("")
                            session.message_length = 0

                            if session.socket is None:
                                del self.sessions[sck]
                            else:
                                self.log.exception(err)
                                self._restart_session(session)
                            continue

                        # We received full payload,
                        # clear session data and parse it.
                        data = session.data
                        session.data = six.b("")
                        session.message_length = 0
                        block_id = struct.unpack('!H', data[0:2])[0]
                        compression = struct.unpack('!B', data[4:5])[0]
                        payload = data[10:]

                        if compression == 0x01:
                            # Data is compressed, uncompress it.
                            payload = zlib.decompress(payload)

                        # Enqueue payload into a callback queue to be
                        # invoked
                        self._callback_pool.queue_callback(session, block_id, payload)
                except select.error as err:
                    # Evaluate sessions if we get a bad file descriptor, if
                    # socket is gone, delete the session.
                    if err.args[0] == errno.EBADF:
                        self._clean_dead_sessions()
                except Exception as err:
                    self.log.exception(err)
        finally:
            for session in self.sessions.values():
                if session is not None:
                    session.stop()