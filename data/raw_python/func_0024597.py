async def stop(self):
        """Stop heartbeat."""
        self.stopped = True
        self.loop_event.set()
        # Waiting for shutdown of loop()
        await self.stopped_event.wait()