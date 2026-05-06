async def async_enqueue_sync(self, func, *func_args):
        '''
        Enqueue an arbitrary synchronous function.
        '''
        worker = self.pick_sticky(0)  # just pick first always
        args = (func,) + func_args
        await worker.enqueue(enums.Task.FUNC, args)