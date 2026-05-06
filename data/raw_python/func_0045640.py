def publish(self, message, key=None, **kws):
        """Put a message in the queue and updates any coroutine wating with
        fetch. *works as a coroutine operation*"""
        return PSPut(self, message, key, **kws)