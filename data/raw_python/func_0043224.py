async def subscribe(self, topic):
        """
        Subscribe the socket to the specified topic.

        :param topic: The topic to subscribe to.
        """
        if self.socket_type not in {SUB, XSUB}:
            raise AssertionError(
                "A %s socket cannot subscribe." % self.socket_type.decode(),
            )

        # Do this **BEFORE** awaiting so that new connections created during
        # the execution below honor the setting.
        self._subscriptions.append(topic)
        tasks = [
            asyncio.ensure_future(
                peer.connection.local_subscribe(topic),
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