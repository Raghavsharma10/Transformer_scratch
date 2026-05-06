def handle_pubrel(self):
        """Handle incoming PUBREL packet."""
        self.logger.info("PUBREL received")

        ret, mid = self.in_packet.read_uint16()

        if ret != NC.ERR_SUCCESS:
            return ret

        evt = event.EventPubrel(mid)
        self.push_event(evt)

        return NC.ERR_SUCCESS