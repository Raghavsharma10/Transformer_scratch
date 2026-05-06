def create_message(self, message_data, delivery_mode=None, priority=None,
                       content_type=None, content_encoding=None,
                       serializer=None):
        """With any data, serialize it and encapsulate it in a AMQP
        message with the proper headers set."""

        delivery_mode = delivery_mode or self.delivery_mode

        # No content_type? Then we're serializing the data internally.
        if not content_type:
            serializer = serializer or self.serializer
            (content_type, content_encoding,
             message_data) = serialization.encode(message_data,
                                                  serializer=serializer)
        else:
            # If the programmer doesn't want us to serialize,
            # make sure content_encoding is set.
            if isinstance(message_data, unicode):
                if not content_encoding:
                    content_encoding = 'utf-8'
                message_data = message_data.encode(content_encoding)

            # If they passed in a string, we can't know anything
            # about it.  So assume it's binary data.
            elif not content_encoding:
                content_encoding = 'binary'

        return self.backend.prepare_message(message_data, delivery_mode,
                                            priority=priority,
                                            content_type=content_type,
                                            content_encoding=content_encoding)