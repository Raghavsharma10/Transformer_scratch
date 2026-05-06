async def _fair_send(self, frames):
        """
        Send from the first available, non-blocking peer or wait until one
        meets the condition.

        :params frames: The frames to write.
        :returns: The peer that was used.
        """
        peer = await self._fair_get_out_peer()
        peer.outbox.write_nowait(frames)
        return peer