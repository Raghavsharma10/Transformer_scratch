def handle_pubrec(self):
        """Handle incoming PUBREC packet."""
        self.logger.info("PUBREC received")

        ret, mid = self.in_packet.read_uint16()

        if ret != NC.ERR_SUCCESS:
            return ret

        evt = event.EventPubrec(mid)
        self.push_event(evt)

        return NC.ERR_SUCCESS