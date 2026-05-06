async def run_async_task(self, container, asynctask, newthread = True):
        "Run asynctask(sender) in task pool, call sender(events) to send customized events, return result"
        e = TaskEvent(self, async_task = asynctask, newthread = newthread)
        await container.wait_for_send(e)
        ev = await TaskDoneEvent.createMatcher(e)
        if hasattr(ev, 'exception'):
            raise ev.exception
        else:
            return ev.result