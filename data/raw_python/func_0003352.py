async def wait_for_all_empty(self, *queues):
        """
        Wait for multiple queues to be empty at the same time.

        Require delegate when calling from coroutines running in other containers
        """
        matchers = [m for m in (q.waitForEmpty() for q in queues) if m is not None]
        while matchers:
            await self.wait_for_all(*matchers)
            matchers = [m for m in (q.waitForEmpty() for q in queues) if m is not None]