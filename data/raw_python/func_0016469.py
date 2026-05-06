def prepare_message(self, message_data, delivery_mode, priority=None,
                content_type=None, content_encoding=None):
        """Encapsulate data into a AMQP message."""
        return amqp.Message(message_data, properties={
                "delivery_mode": delivery_mode,
                "priority": priority,
                "content_type": content_type,
                "content_encoding": content_encoding})