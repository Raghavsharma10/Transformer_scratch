def _handle_heartbeat(self, sender, data):
        """
        Handles a raw heart beat

        :param sender: Sender (address, port) tuple
        :param data: Raw packet data
        """
        # Format of packet
        parsed, data = self._unpack("<B", data)
        format = parsed[0]
        if format == PACKET_FORMAT_VERSION:
            # Kind of beat
            parsed, data = self._unpack("<B", data)
            kind = parsed[0]
            if kind == PACKET_TYPE_HEARTBEAT:
                # Extract content
                parsed, data = self._unpack("<H", data)
                port = parsed[0]
                path, data = self._unpack_string(data)
                uid, data = self._unpack_string(data)
                node_uid, data = self._unpack_string(data)
                try:
                    app_id, data = self._unpack_string(data)
                except struct.error:
                    # Compatibility with previous version
                    app_id = herald.DEFAULT_APPLICATION_ID

            elif kind == PACKET_TYPE_LASTBEAT:
                # Peer is going away
                uid, data = self._unpack_string(data)
                app_id, data = self._unpack_string(data)
                port = -1
                path = None
                node_uid = None

            else:
                _logger.warning("Unknown kind of packet: %d", kind)
                return

            try:
                self._callback(kind, uid, node_uid, app_id, sender[0], port, path)
            except Exception as ex:
                _logger.exception("Error handling heart beat: %s", ex)