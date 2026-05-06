async def unsubscribe(self, topic):
        """
        Unsubscribe the socket from the specified topic.

        :param topic: The topic to unsubscribe from.
        """
        if self.socket_type not in {SUB, XSUB}:
            raise AssertionError(
                "A %s socket cannot unsubscribe." % self.socket_type.decode(),
            )

        # Do this **BEFORE** awaiting so that new connections created during
        # the execution below honor the setting.
        self._subscriptions.remove(topic)
        tasks = [
            asyncio.ensure_future(
                peer.connection.local_unsubscribe(topic),
                loop=self.loop,
            )
            for peer in self._peers
            if peer.connection
        ]

        if tasks:
            try:
                await asyncio.wait(tasks, loop=self.loop)
            finally:
                for task in tasks:
                    task.cancel()