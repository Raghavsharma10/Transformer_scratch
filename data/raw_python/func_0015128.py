def handle_connack(self):
        """Handle incoming CONNACK command."""
        self.logger.info("CONNACK reveived")
        ret, flags = self.in_packet.read_byte()
        if ret != NC.ERR_SUCCESS:
            self.logger.error("error read byte")
            return ret
        
        # useful for v3.1.1 only
        session_present = flags & 0x01

        ret, retcode = self.in_packet.read_byte()
        if ret != NC.ERR_SUCCESS:
            return ret
        
        evt = event.EventConnack(retcode, session_present)
        self.push_event(evt)
        
        if retcode == NC.CONNECT_ACCEPTED:
            self.state = NC.CS_CONNECTED
            return NC.ERR_SUCCESS
        
        elif retcode >= 1 and retcode <= 5:
            return NC.ERR_CONN_REFUSED
        else:
            return NC.ERR_PROTOCOL