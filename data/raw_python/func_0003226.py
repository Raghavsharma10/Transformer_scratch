async def run_gen_task(self, container, gentask, newthread = True):
        "Run generator gentask() in task pool, yield customized events"
        e = TaskEvent(self, gen_task = gentask, newthread = newthread)
        await container.wait_for_send(e)
        ev = await TaskDoneEvent.createMatcher(e)
        if hasattr(ev, 'exception'):
            raise ev.exception
        else:
            return ev.result