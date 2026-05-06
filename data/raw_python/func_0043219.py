async def _wait_peers(self):
        """
        Blocks until at least one non-dead peer is available.
        """
        # Make sure we remove dead peers.
        for p in self._peers[:]:
            if p.dead:
                self._peers.remove(p)

        while not self._peers:
            await self._peers.wait_not_empty()