async def recv_multipart(self):
        """
        Read from all the associated sockets.

        :returns: A list of tuples (socket, frames) for each socket that
            returned a result.
        """
        if not self._sockets:
            return []

        results = []

        async def recv_and_store(socket):
            frames = await socket.recv_multipart()
            results.append((socket, frames))

        tasks = [
            asyncio.ensure_future(recv_and_store(socket), loop=self.loop)
            for socket in self._sockets
        ]

        try:
            await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
                loop=self.loop,
            )
        finally:
            for task in tasks:
                task.cancel()

        return results