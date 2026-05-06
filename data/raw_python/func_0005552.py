def worker_allocator(self, async_loop, to_do, **kwargs):
        """ Handler starting the asyncio part. """
        d = kwargs
        threading.Thread(
            target=self._asyncio_thread, args=(async_loop, to_do, d)
        ).start()