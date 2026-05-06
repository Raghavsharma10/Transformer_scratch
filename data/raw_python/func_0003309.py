def header(self, key, value, replace = True):
        "Send a new header"
        if hasattr(key, 'encode'):
            key = key.encode('ascii')
        if hasattr(value, 'encode'):
            value = value.encode(self.encoding)
        if replace:
            self.sent_headers = [(k,v) for k,v in self.sent_headers if k.lower() != key.lower()]
        self.sent_headers.append((key, value))