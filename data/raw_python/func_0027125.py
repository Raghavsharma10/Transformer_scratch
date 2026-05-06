def encrypt(self, message):
        """ Encrypt the given message """

        if not isinstance(message, (bytes, str)):
            raise TypeError
        
        return hashlib.sha1(message.encode('utf-8')).hexdigest()