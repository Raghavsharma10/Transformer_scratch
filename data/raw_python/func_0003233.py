async def destroy(self, container = None):
        """
        Destroy the created subqueue to change the behavior back to Lock
        """
        if container is None:
            container = RoutineContainer(self.scheduler)
        if self.queue is not None:
            await container.syscall_noreturn(syscall_removequeue(self.scheduler.queue, self.queue))
            self.queue = None