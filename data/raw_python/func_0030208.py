def slice(self, items):
        '''Slice the sequence of all items to obtain them for current page.'''
        if self.limit:
            if self.page>self.pages_count:
                return []
            if self.page == self.pages_count:
                return items[self.limit*(self.page-1):]
            return items[self.limit*(self.page-1):self.limit*self.page]
        else:
            return items[:]