def enqueue_download(self, resource):
        '''
        Enqueue the download of the given foreign resource.

        Deprecated: Use async version instead
        '''
        worker = self.pick_sticky(resource.url_string)
        coro = worker.enqueue(enums.Task.DOWNLOAD, (resource,))
        asyncio.ensure_future(coro)