def handle_pingresp(self):
        """Handle incoming PINGRESP packet."""
        self.logger.debug("PINGRESP received")
        self.push_event(event.EventPingResp())
        return NC.ERR_SUCCESS