async def _fair_get_in_peer(self):
        """
        Get the first available available inbound peer in a fair manner.

        :returns: A `Peer` inbox, whose inbox is guaranteed not to be
            empty (and thus can be read from without blocking).
        """
        peer = None

        while not peer:
            await self._wait_peers()

            # This rotates the list, implementing fair-queuing.
            peers = list(self._in_peers)

            tasks = [asyncio.ensure_future(self._in_peers.wait_change())]
            tasks.extend([
                asyncio.ensure_future(
                    p.inbox.wait_not_empty(),
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
            peer = next(
                (
                    p
                    for task, p in zip(tasks, peers)
                    if task in done and not task.cancelled()
                ),
                None,
            )

        return peer