def enqueue_sync(self, func, *func_args):
        '''
        Enqueue an arbitrary synchronous function.

        Deprecated: Use async version instead
        '''
        worker = self.pick_sticky(0)  # just pick first always
        args = (func,) + func_args
        coro = worker.enqueue(enums.Task.FUNC, args)
        asyncio.ensure_future(coro)