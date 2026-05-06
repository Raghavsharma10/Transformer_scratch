def run(self, *args, **kwargs):
        """This runs in a greenlet"""
        return_value = self.coro(*args, **kwargs)

        # i don't like this but greenlets are so dodgy i have no other choice
        raise StopIteration(return_value)