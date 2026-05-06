async def run_task(self, container, task, newthread = False):
        "Run task() in task pool. Raise an exception or return the return value"
        e = TaskEvent(self, task=task, newthread = newthread)
        await container.wait_for_send(e)
        ev = await TaskDoneEvent.createMatcher(e)
        if hasattr(ev, 'exception'):
            raise ev.exception
        else:
            return ev.result