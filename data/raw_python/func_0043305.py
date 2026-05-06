async def await_until_closing(self, coro):
        """
        Wait for some task to complete but aborts as soon asthe instance is
        being closed.

        :param coro: The coroutine or future-like object to wait for.
        """
        wait_task = asyncio.ensure_future(self.wait_closing(), loop=self.loop)
        coro_task = asyncio.ensure_future(coro, loop=self.loop)

        try:
            done, pending = await asyncio.wait(
                [wait_task, coro_task],
                return_when=asyncio.FIRST_COMPLETED,
                loop=self.loop,
            )

        finally:
            wait_task.cancel()
            coro_task.cancel()

        # It could be that the previous instructions cancelled coro_task if it
        # wasn't done yet.
        return await coro_task