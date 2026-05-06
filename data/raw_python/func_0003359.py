async def delegate_other(self, subprocess, container, retnames = ('',), forceclose = False):
        '''
        DEPRECATED Another format of delegate allows delegate a subprocess in another container, and get some returning values
        the subprocess is actually running in 'container'. ::
        
            ret = await self.delegate_other(c.method(), c)
        
        :return: a tuple for retnames values
        
        '''
        finish, r = self.beginDelegateOther(subprocess, container, retnames)
        return await self.end_delegate(finish, r, forceclose)