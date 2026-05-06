def replay_checker(self, callback):
        """
        Decorate a method that receives the request headers and returns a bool
        indicating whether we should proceed with the request. This can be used
        to protect against replay attacks. For example, this method could check
        the request date header value is within a delta value of the server time.
        """
        if not callback or not callable(callback):
            raise Exception("Please pass in a callable that protects against replays")
        self.replay_checker_callback = callback
        return callback