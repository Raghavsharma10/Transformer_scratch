async def lock(self, container = None):
        "Wait for lock acquire"
        if container is None:
            container = RoutineContainer.get_container(self.scheduler)
        if self.locked:
            pass
        elif self.lockroutine:
            await LockedEvent.createMatcher(self)
        else:
            await container.wait_for_send(LockEvent(self.context, self.key, self))
            self.locked = True