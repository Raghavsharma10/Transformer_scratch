def clear(self, transport):
        """Issues a device clear command."""
        logger.debug('IEC60488 clear')
        with transport:
            try:
                transport.clear()
            except AttributeError:
                clear_msg = self.create_message('*CLS')
                transport.write(clear_msg)