async def async_enqueue_multiconvert(self, url_string, to_type):
        '''
        Enqueue a multi-step conversion process, from the given URL string
        (which is assumed to have been downloaded / resolved)
        '''
        worker = self.pick_sticky(url_string)
        args = (url_string, to_type)
        await worker.enqueue(enums.Task.MULTICONVERT, args)