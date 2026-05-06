def handle_puback(self):
        """Handle incoming PUBACK packet."""
        self.logger.info("PUBACK received")

        ret, mid = self.in_packet.read_uint16()

        if ret != NC.ERR_SUCCESS:
            return ret

        evt = event.EventPuback(mid)
        self.push_event(evt)

        return NC.ERR_SUCCESS