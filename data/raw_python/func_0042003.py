def context_loader(self, callback):
        """
        Decorate a method that receives a key id and returns an object or dict
        that will be available in the request context as g.cavage_context
        """
        if not callback or not callable(callback):
            raise Exception("Please pass in a callable that loads your context.")
        self.context_loader_callback = callback
        return callback