async def restart_walk(self):
        """
        Force a re-walk
        """
        if not self._restartwalk:
            self._restartwalk = True
            await self.wait_for_send(FlowUpdaterNotification(self, FlowUpdaterNotification.STARTWALK))