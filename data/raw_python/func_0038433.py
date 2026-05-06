def is_type(msg_type, msg):
        """
        Return message's type is or not
        """
        for prop in MessageType.FIELDS[msg_type]["must"]:
            if msg.get(prop, False) is False:
                return False
        for prop in MessageType.FIELDS[msg_type]["prohibit"]:
            if msg.get(prop, False) is not False:
                return False

        return True