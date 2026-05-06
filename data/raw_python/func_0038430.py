def to_response(self, sign, code=200, data=None):
        """
        transform message to response message
        Notice: this method will return a deepcopy
        """
        msg = copy.deepcopy(self)
        msg.data = data

        setattr(msg, 'code', code)
        for _ in ["query", "param", "tunnel"]:
            if not hasattr(msg, _):
                continue
            delattr(msg, _)

        if hasattr(msg, 'sign') and isinstance(msg.sign, list):
            msg.sign.append(sign)
        else:
            msg.sign = [sign]

        msg._type = Message.get_message_type(msg.__dict__)

        return msg