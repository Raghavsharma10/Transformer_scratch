def trigger(self, transport):
        """Triggers the transport."""
        logger.debug('IEC60488 trigger')
        with transport:
            try:
                transport.trigger()
            except AttributeError:
                trigger_msg = self.create_message('*TRG')
                transport.write(trigger_msg)