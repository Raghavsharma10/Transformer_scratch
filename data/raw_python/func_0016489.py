def decode(self):
        """Deserialize the message body, returning the original
        python structure sent by the publisher."""
        return serialization.decode(self.body, self.content_type,
                                    self.content_encoding)