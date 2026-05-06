async def run_multiconvert(self, url_string, to_type):
        '''
        Enqueues in succession all conversions steps necessary to take the
        given URL and convert it to to_type, storing the result in the cache
        '''
        async def enq_convert(*args):
            await self.enqueue(Task.CONVERT, args)
        await tasks.multiconvert(url_string, to_type, enq_convert)