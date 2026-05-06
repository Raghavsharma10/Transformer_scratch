def build_payload(self, tag, message):
        """ Encode, sign payload(optional) and attach subscription tag """
        message = self.encode(message)
        message = self.sign(message)
        payload = bytes(tag.encode('utf-8')) + message
        return payload