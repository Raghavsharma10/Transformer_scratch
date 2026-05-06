def secret_loader(self, callback):
        """
        Decorate a method that receives a key id and returns a secret key
        """
        if not callback or not callable(callback):
            raise Exception("Please pass in a callable that loads secret keys")
        self.secret_loader_callback = callback
        return callback