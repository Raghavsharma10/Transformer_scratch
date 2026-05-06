async def _fair_recv(self):
        """
        Receive from all the existing peers, rotating the list of peers every
        time.

        :returns: The frames.
        """
        with await self._read_lock:
            peer = await self._fair_get_in_peer()
            result = peer.inbox.read_nowait()

        return result