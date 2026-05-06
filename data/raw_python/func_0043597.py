async def wait_change(self):
        """
        Wait for the list to change.
        """
        future = asyncio.Future(loop=self.loop)
        self._change_futures.add(future)
        future.add_done_callback(self._change_futures.discard)
        await future