def unblock(self, event):
        '''
        Remove a block 
        '''
        if event not in self.blockEvents:
            return
        self.blockEvents[event].unblock(event)
        del self.blockEvents[event]