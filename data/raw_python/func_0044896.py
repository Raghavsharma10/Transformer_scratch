def add(self, coro, args=(), kwargs={}, first=True):
        """Add a coroutine in the scheduler. You can add arguments
        (_args_, _kwargs_) to init the coroutine with."""
        assert callable(coro), "'%s' not a callable object" % coro
        coro = coro(*args, **kwargs)
        if first:
            self.active.append( (None, coro) )
        else:
            self.active.appendleft( (None, coro) )
        return coro