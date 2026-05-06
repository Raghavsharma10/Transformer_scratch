def remove(self, toRemove):
        '''
            remove - Remove an item from this tag collection

            @param toRemove - an AdvancedTag
        '''
        list.remove(self, toRemove)
        self.uids.remove(toRemove.uid)