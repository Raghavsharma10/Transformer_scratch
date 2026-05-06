def set_handler(self, handler, start_heartbeat=True):
        """ Set active handler for the session

        @param handler: Associate active cyclone handler with the session

        @param start_heartbeat: Should session start heartbeat immediately
        """
        # Check if session already has associated handler
        if self.handler is not None:
            handler.send_pack(proto.disconnect(2010,
                                               "Another connection still open"))
            return False

        if self._verify_ip and self.conn_info is not None:
            # If IP address doesn't match - refuse connection
            if handler.request.remote_ip != self.conn_info.ip:
                log.msg('Attempted to attach to session %s (%s) from '
                        'different IP (%s)' % ( self.session_id,
                                                self.conn_info.ip,
                                                handler.request.remote_ip
                                              )
                       )

                handler.send_pack(proto.disconnect(2010, 'Attempted to connect '
                                         'to session from different IP'))
                return False

        if ( self.state == SESSION_STATE.CLOSING
             or self.state == SESSION_STATE.CLOSED):
            handler.send_pack(proto.disconnect(*self.get_close_reason()))
            return False

        # Associate handler and promote session
        super(Session, self).set_handler(handler)

        self.promote()

        if start_heartbeat:
            self.start_heartbeat()

        return True