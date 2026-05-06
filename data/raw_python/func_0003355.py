async def delegate(self, subprocess, forceclose = False):
        '''
        Run a subprocess without container support
        
        Many subprocess assume itself running in a specified container, it uses container reference
        like self.events. Calling the subprocess in other containers will fail.
        
        With delegate, you can call a subprocess in any container (or without a container)::
        
            r = await c.delegate(c.someprocess())
        
        :return: original return value
        '''
        finish, r = self.begin_delegate(subprocess)
        return await self.end_delegate(finish, r, forceclose)