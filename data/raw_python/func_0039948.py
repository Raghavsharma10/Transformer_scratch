def type(self):
        '''returns kind of stim ("column" or "times"), based on what parameters are set'''
        if self.column!=None or self.column_file:
            return "column"
        if self.times!=None or self.times_file:
            return "times"
        return None