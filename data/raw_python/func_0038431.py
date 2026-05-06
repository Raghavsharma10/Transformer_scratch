def to_event(self):
        """
        get rid of id, sign, tunnel and update message type
        Notice: this method will return a deepcopy
        """
        msg = copy.deepcopy(self)
        for _ in ["id", "sign", "tunnel", "query", "param"]:
            if not hasattr(msg, _):
                continue
            delattr(msg, _)

        msg._type = Message.get_message_type(msg.__dict__)

        return msg