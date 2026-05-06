def _receive_callback(self, raw_message):
        """Internal method used when a message is received in consume mode."""
        message = self.backend.message_to_python(raw_message)

        if self.auto_ack and not message.acknowledged:
            message.ack()
        self.receive(message.payload, message)