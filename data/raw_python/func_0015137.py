def handle_pubcomp(self):
        """Handle incoming PUBCOMP packet."""
        self.logger.info("PUBCOMP received")

        ret, mid = self.in_packet.read_uint16()

        if ret != NC.ERR_SUCCESS:
            return ret

        evt = event.EventPubcomp(mid)
        self.push_event(evt)

        return NC.ERR_SUCCESS