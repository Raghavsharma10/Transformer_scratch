def enqueue_convert(self, converter, from_resource, to_resource):
        '''
        Enqueue use of the given converter to convert to given
        resources.

        Deprecated: Use async version instead
        '''
        worker = self.pick_sticky(from_resource.url_string)
        args = (converter, from_resource, to_resource)
        coro = worker.enqueue(enums.Task.CONVERT, args)
        asyncio.ensure_future(coro)