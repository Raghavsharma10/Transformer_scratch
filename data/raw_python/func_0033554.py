async def async_enqueue_convert(self, converter, from_, to):
        '''
        Enqueue use of the given converter to convert to given
        from and to resources.
        '''
        worker = self.pick_sticky(from_.url_string)
        args = (converter, from_, to)
        await worker.enqueue(enums.Task.CONVERT, args)