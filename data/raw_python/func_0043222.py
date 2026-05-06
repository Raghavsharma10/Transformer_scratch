async def _fair_get_out_peer(self):
        """
        Get the first available peer, with non-blocking inbox or wait until one
        meets the condition.

        :returns: The peer whose outbox is ready to be written to.
        """
        peer = None

        while not peer:
            await self._wait_peers()

            # This rotates the list, implementing fair-queuing.
            peers = list(self._out_peers)

            tasks = [asyncio.ensure_future(self._out_peers.wait_change())]
            tasks.extend([
                asyncio.ensure_future(
                    p.outbox.wait_not_full(),
                    loop=self.loop,
                )
                for p in peers
            ])

            try:
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                    loop=self.loop,
                )
            finally:
                for task in tasks:
                    task.cancel()

            tasks.pop(0)  # pop the wait_change task.
            peer = next(  # pragma: no cover
                (
                    p
                    for task, p in zip(tasks, peers)
                    if task in done and not p.outbox.full()
                ),
                None,
            )

        return peer