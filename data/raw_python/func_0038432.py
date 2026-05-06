def get_message_type(message):
        """
        Return message's type
        """
        for msg_type in MessageType.FIELDS:
            if Message.is_type(msg_type, message):
                return msg_type

        return MessageType.UNKNOWN